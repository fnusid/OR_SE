
import torch 
import torch.nn as nn
import numpy as np



class STFT(nn.Module):
    def __init__(self,
                 frame_len = 512,
                 hop_len = 256,
                 fft_len = 512,
                 window_type='sine',
                 trainable_window=False,
                 return_complex= True,
                 ):
        super(STFT, self).__init__()
        self.frame_len = frame_len
        self.hop_len = hop_len
        self.fft_len = fft_len
        self.window_type = window_type
        self.trainable_window = trainable_window
        self.return_complex = return_complex
        
        if self.window_type == 'sine':
            win = np.sin((np.arange(0.5, frame_len - 0.5 + 1)) / frame_len * np.pi)
        elif self.window_type == 'hanning':
            win = np.hanning(frame_len)
        elif self.window_type == 'hamming':
            win = np.hamming(frame_len)
        elif self.window_type == 'blackman':
            win = np.blackman(frame_len)
        win = torch.from_numpy(win).float()
        if self.trainable_window:
            self.win = nn.Parameter(win)
        else:
            self.register_buffer('win', win)

    def forward(self, x):
        '''
        x: (B, T)
        '''
        x_stft = torch.stft(x, return_complex=False, n_fft=self.fft_len, hop_length=self.hop_len, win_length=self.frame_len, window=self.win, )
        x_stft = x_stft.permute(0,3,1,2)  # [B, 2, F, T]

        if self.return_complex:
            return x_stft  # [B, 2, F, T], complex tensor
        else:
            real, imag = x_stft.unbind(1)  # [B, 2, F, T]
            mag = torch.sqrt(real**2 + imag**2)
            phase = torch.atan2(imag, real)
            return torch.stack([mag, phase], dim=1)  # [B, 2, F, T]


class ISTFT(nn.Module):
    def __init__(self,
                 frame_len = 512,
                 hop_len = 256,
                 fft_len = 512,
                 window_type='sine',
                 trainable_window=False,
                 length = 1,
                 ):
        super(ISTFT, self).__init__()
        self.frame_len = frame_len
        self.hop_len = hop_len
        self.fft_len = fft_len
        self.window_type = window_type
        self.trainable_window = trainable_window
        self.length = int(length*16000)
        
        if self.window_type == 'sine':
            win = np.sin((np.arange(0.5, frame_len - 0.5 + 1)) / frame_len * np.pi)
        elif self.window_type == 'hanning':
            win = np.hanning(frame_len)
        elif self.window_type == 'hamming':
            win = np.hamming(frame_len)
        elif self.window_type == 'blackman':
            win = np.blackman(frame_len)
        win = torch.from_numpy(win).float()
        if self.trainable_window:
            self.win = nn.Parameter(win)
        else:
            self.register_buffer('win', win)

    def forward(self, x_stft):
        '''
        x_stft: (B, 2, F, T), complex tensor
        '''
        if x_stft.size(1) == 2:
            real, imag = x_stft.unbind(1)  # [B, F, T]
            x_stft = torch.complex(real, imag)  # [B, F, T]
        x = torch.istft(x_stft, n_fft=self.fft_len, hop_length=self.hop_len, win_length=self.frame_len, window=self.win, length=self.length)
        return x
