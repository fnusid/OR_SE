import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
import wandb
import sys
from src.loss import LossWrapper
from src.metric import MetricWrapper
from utils.stft_istft import STFT, ISTFT







class ORSEModel(pl.LightningModule):
    def __init__(self,
                 frame_len = 512,
                 hop_len = 256,
                 fft_len = 512,
                 win = 'sine',
                 length = 1,
                 enc_filters = [16,32,64,64,128,256],
                 enc_kernels = [(5,1), (3,1), (3,1), (3,1), (3,1), (3,1)], #[F,T] keeping it causal
                 enc_strides = [(2,1), (2,1), (2,1), (2,1), (2,1), (2,1)],
                 in_channels = 2, # real and imag, or mag and phase
                 num_bottleneck_layers = 2,
                 loss_fn = ['mse', 'complex_spectral'],
                 loss_weights = [0.5, 0.5],
                 lr = 1e-3,
                 alpha=0.5, 
                 metric='DNSMOS',

                 ):
        super(ORSEModel, self).__init__()

        # Define the hyperparameters
        self.loss_fn = LossWrapper(loss_fn, loss_weights, alpha=alpha)
        self.metric = MetricWrapper(metric)
        self.lr = lr

        #Define the stft params and layers
        self.frame_len = frame_len
        self.hop_len = hop_len
        self.fft_len = fft_len
        self.win = win
        self.length = length
        self.stft = STFT(frame_len=self.frame_len, hop_len=self.hop_len, fft_len=self.fft_len, window_type=self.win, return_complex=True)
        self.istft = ISTFT(frame_len=self.frame_len, hop_len=self.hop_len, fft_len=self.fft_len, window_type=self.win, length=self.length)

    
        #Define the encoder layer params
        self.enc_filters = enc_filters
        self.enc_kernels = enc_kernels #[F,T] keeping it causal
        self.enc_strides = enc_strides
        self.in_channels = in_channels # real and imag, or mag and phase
        self.num_bottleneck_layers = num_bottleneck_layers

        #define encoder layer by creating alternate function later on
        self.enc_layers = nn.ModuleList()
        for i in range(len(self.enc_filters)):
            out_channels = self.enc_filters[i]
            kernel_size = self.enc_kernels[i]
            stride = self.enc_strides[i]
            enc_layer = self.build_enc_layer(self.in_channels, out_channels, kernel_size, stride)
            self.enc_layers.append(enc_layer)
            self.in_channels = out_channels

        #define the bottleneck layer the same way
        self.bottleneck_layers = nn.ModuleList()
        for i in range(self.num_bottleneck_layers):
            bottleneck_layer = self.build_bottleneck_layer(self.in_channels*5)
            self.bottleneck_layers.append(bottleneck_layer)

        # define the decoder layer
        self.dec_layers = nn.ModuleList()
        for i in range(len(self.enc_filters)-1, -1, -1):
            out_channels = self.enc_filters[i-1] if i > 0 else 2
            kernel_size = self.enc_kernels[i]
            stride = self.enc_strides[i]
            dec_layer = self.build_dec_layer(self.in_channels, out_channels, kernel_size, stride)
            self.dec_layers.append(dec_layer)
            self.in_channels = out_channels

    def forward(self, x):
        '''
        x: (B, T)
        '''
        x_stft = self.stft(x) # [B, 2, F, T]
        out = x_stft
        #Encoder
        enc_out_list = []
        for enc_layer in self.enc_layers:
            out = enc_layer(out)
            enc_out_list.append(out)
        #Bottleneck
        B, C, F, T = out.size()
        out = out.permute(0,3,1,2).contiguous().view(B,T,C*F) # [B, T, C*F]
        for bottleneck_layer in self.bottleneck_layers:
            out, _ = bottleneck_layer[0](out)
            out = out.permute(0,2,1).contiguous() # [B, C*F, T]
            out = bottleneck_layer[1:](out)
            out = out.permute(0,2,1).contiguous() # [B, T, C*F]
        out = out.view(B,T,C,F).permute(0,2,3,1).contiguous() # [B, C, F, T]
        #Decoder
        for dec_layer in self.dec_layers:
            out += enc_out_list.pop() #skip connection
            out = dec_layer(out)
        
        #define a complex ratio masking function here
        mask = torch.tanh(out) # [B, 2, F, T], range -1 to 1
        real, imag = x_stft.unbind(1)
        out = torch.stack([real*mask[:,0], imag*mask[:,1]], dim=1) # [B, 2, F, T]

        x_stft_est = out
        x_est = self.istft(x_stft_est) # [B, T]
        return x_est, x_stft_est

    def build_enc_layer(self, in_channels, out_channels, kernel_size, stride):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size[0]//2,0)), #causal conv
                              nn.BatchNorm2d(out_channels),
                                nn.ReLU())
    
    def build_bottleneck_layer(self, channels):
        bl = nn.Sequential(nn.LSTM(input_size=channels, hidden_size=channels, num_layers=1, batch_first=True, bidirectional=False),
                            nn.BatchNorm1d(channels),
                            nn.ReLU())
        return bl
    
    def build_dec_layer(self, in_channels, out_channels, kernel_size, stride):
        return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size[0]//2,0)),
                              nn.BatchNorm2d(out_channels),
                                nn.ReLU())
    
    def on_train_epoch_end(self):
        pass

    def on_valid_epoch_end(self):
        pass

    def training_step(self, batch, batch_idx):
        noisy, clean = batch
        loss, enhanced_time = self._common_step(noisy)
        self.log_dict({'train_loss': loss, }, prog_bar=True, on_step=True, on_epoch=True)
        return {'loss': loss,}
    
    def validation_step(self, batch, batch_idx):
        noisy, clean = batch
        loss, enhanced_time = self._common_step(noisy, clean)
        metric_scores = self.metric(enhanced_time, sr=16000)
        self.log_dict({'val_loss': loss, 'SIG': metric_scores['SIG'], 'BAK': metric_scores['BAK'], 'OVRL': metric_scores['OVRL']}, prog_bar=True, on_step=False, on_epoch=True)
        return {'val_loss': loss, 'SIG': metric_scores['SIG'], 'BAK': metric_scores['BAK'], 'OVRL': metric_scores['OVRL']}
    
    def _common_step(self, noisy, clean):
        enhanced_time, enhanced_spec = self.forward(noisy)
        clean_spec = self.stft(clean)
        loss = self.loss_fn(enhanced_time, enhanced_spec, clean, clean_spec)
        return loss, enhanced_time
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


    
    

        
if __name__ == "__main__":
    model = ORSEModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    x = torch.randn(2,16000).to(device)
    y = model(x)
    print(y.shape)


    