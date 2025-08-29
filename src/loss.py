import torch
import torch.nn as nn
import torch.nn.functional as F



class LossWrapper(nn.Module):
    def __init__(self, loss_fn_list, loss_weights, alpha=0.5):
        super(LossWrapper, self).__init__()
        self.loss_fn_list = loss_fn_list
        self.loss_weights = loss_weights
        self.loss_fns = []
        for loss_fn in self.loss_fn_list:
            if loss_fn == 'mse':
                self.loss_fns.append(nn.MSELoss())
            elif loss_fn == 'l1':
                self.loss_fns.append(nn.L1Loss())
            elif loss_fn == 'complex_spectral':
                self.loss_fns.append(self.complex_spectral_loss)
            else:
                raise ValueError(f"Unsupported loss function: {loss_fn}")
            
    def forward(self, est, est_spec, target, target_spec=None):
        total_loss = 0
        for i, loss_fn in enumerate(self.loss_fns):
            if self.loss_fn_list[i] == 'complex_spectral':
                if target_spec is None:
                    raise ValueError("target_spec must be provided for complex_spectral loss")
                else:
                    loss = loss_fn(est, target_spec)
            else:
                loss = loss_fn(est, target)
            total_loss += self.loss_weights[i] * loss
        return total_loss
    

    def complex_spectral_loss(self, est, target, alpha=0.5):
        '''
        est, target: (B, 2, F, T), complex tensor
        '''
        real_est, imag_est = est.unbind(1)
        real_target, imag_target = target.unbind(1)
        mag_est = torch.sqrt(real_est**2 + imag_est**2)
        mag_target = torch.sqrt(real_target**2 + imag_target**2)
        phase_est = torch.atan2(imag_est, real_est)
        phase_target = torch.atan2(imag_target, real_target)
        mag_loss = F.mse_loss(mag_est, mag_target)
        phase_loss = F.mse_loss(phase_est, phase_target)
        return alpha * mag_loss + (1-alpha) * phase_loss