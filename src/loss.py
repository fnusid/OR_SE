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
                    loss = loss_fn(est_spec, target_spec)
            else:
                loss = loss_fn(est, target)
            total_loss = total_loss +  self.loss_weights[i] * loss
        return total_loss
    

    def masked_mean(self, x, mask, eps=1e-8):
        # x, mask: same shape
        return (x * mask).sum() / (mask.sum().clamp(min=1.0))

    def complex_spectral_loss(self, est, target, alpha=0.5, eps=1e-8, mask_thresh=1e-4, weight_by_mag=True):
        """
        est, target: (B, 2, F, T) with channels [real, imag]
        alpha: weight for magnitude term; (1-alpha) for phase term
        """
        # Split real/imag
        er, ei = est.unbind(dim=1)      # (B, F, T)
        tr, ti = target.unbind(dim=1)   # (B, F, T)

        # Magnitudes (stable)
        emag = torch.sqrt(er*er + ei*ei + eps)
        tmag = torch.sqrt(tr*tr + ti*ti + eps)

        # ----- magnitude loss (robust) -----
        # Smooth L1 is more stable than MSE when there are outliers
        mag_l = F.smooth_l1_loss(emag, tmag, reduction='none')  # (B, F, T)

        # ----- phase loss without atan2 -----
        # cos(Δφ) = (Re_e Re_t + Im_e Im_t) / (|E||T|)
        dot = er*tr + ei*ti
        denom = (emag * tmag).clamp_min(eps)
        cos_dphi = (dot / denom).clamp(-1.0, 1.0)
        # 1 - cos(Δφ) ≈ 0 when phases match; smooth & bounded in [0,2]
        phase_l = 1.0 - cos_dphi  # (B, F, T)

        # ----- mask out near-silent bins to avoid noisy gradients -----
        mask = ((emag > mask_thresh) | (tmag > mask_thresh)).float()

        # optional: weight phase loss by target magnitude so strong bins matter more
        if weight_by_mag:
            w = (tmag / (tmag.mean(dim=(1,2), keepdim=True) + eps)).detach()
            phase_l = phase_l * w
            mag_l = mag_l * w

        mag_loss = self.masked_mean(mag_l, mask, eps)
        phase_loss = self.masked_mean(phase_l, mask, eps)

        return alpha * mag_loss + (1.0 - alpha) * phase_loss
