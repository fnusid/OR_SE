import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchaudio
import pytorch_lightning as pl
import numpy as np
import random
from typing import Optional, Any

class ORDataset(Dataset):
    def __init__(self, speech_list, noise_list, rir_list, batch_size=8, num_workers=4, pin_memory=True, **kwargs):
        super(ORDataset, self).__init__()
        self.speech_list = speech_list
        self.noise_list = noise_list
        self.rir_list = rir_list
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.global_snr_range = kwargs.get('global_snr', [-5, 10]) #dB
        self.sr = kwargs.get('sr', 16000)
        self.segment_length = kwargs.get('segment_length', 8)


        #Load all the files 
        with open(self.speech_list) as f: self.speech_files = [line.strip() for line in f if line.strip()]
        with open(self.noise_list) as f: self.noise_files = [line.strip() for line in f if line.strip()]
        with open(self.rir_list) as f: self.rir_files = [line.strip() for line in f if line.strip()]


        #shuffle the files
        random.shuffle(self.speech_files)
        random.shuffle(self.noise_files)
        random.shuffle(self.rir_files)

        #split into train, and val
        num_speech = len(self.speech_files)
        num_noise = len(self.noise_files)
        num_rir = len(self.rir_files)
        
        self.train_speech = self.speech_files[:int(0.8*num_speech)]
        self.val_speech = self.speech_files[int(0.8*num_speech):]

        self.train_noise = self.noise_files[:int(0.8*num_noise)]
        self.val_noise = self.noise_files[int(0.8*num_noise):]
        
        self.train_rir = self.rir_files[:int(0.8*num_rir)]
        self.val_rir = self.rir_files[int(0.8*num_rir):]




    def get_early_reverbs(self, rir, sr, duration=0.05):
        """Extract early reverberation from RIR.
        Args:
            rir (torch.Tensor): RIR tensor of shape (num_mics, rir_length).
            sr (int): Sampling rate.
            duration (float): Duration in seconds to extract early reverberation.
        Returns:
            torch.Tensor: Early reverberation tensor of shape (num_mics, early_rir_length).
        """
        offset = torch.where(rir)[0][0]  # Find the first non-zero index
        rir_resampled = rir[offset:]  # Remove leading zeros
        #pad to original length
        rir_resampled = F.pad(rir_resampled, (0, rir.shape[-1] - rir_resampled.shape[-1]))
        early_rir_length = int(sr * duration)
        rir_desired = rir_resampled[:early_rir_length]
        rir_desired_oglen = F.pad(rir_desired, (0, rir.shape[-1] - rir_desired.shape[-1]))
        return rir_desired_oglen
    
    def apply_reverb(self, speech, rir):
        """Apply reverberation to speech using RIR.
        Args:
            speech (torch.Tensor): Speech tensor of shape (num_samples,).
            rir (torch.Tensor): RIR tensor of shape (num_mics, rir_length).
        Returns:
            torch.Tensor: Reverberated speech tensor of shape (num_mics, num_samples + rir_length - 1).
        """
        if rir.dim() != 1:
            rir = rir[0, :]
        if speech.dim() != 1:
            speech = speech[0, :]

        reverberated_speech = F.conv1d(speech.unsqueeze(0).unsqueeze(0), rir.unsqueeze(0).unsqueeze(0), padding=rir.shape[-1]-1)
        reverberated_speech = reverberated_speech.squeeze(0).squeeze(0)
        reverberated_speech = reverberated_speech[:len(speech)]
        return reverberated_speech
    
    def add_noise(self, speech, noise, snr_db):
        """Add noise to speech at a specified SNR.
        Args:
            speech (torch.Tensor): Speech tensor of shape (num_samples,).
            noise (torch.Tensor): Noise tensor of shape (num_samples,).
            snr_db (float): Desired SNR in dB.
        Returns:
            torch.Tensor: Noisy speech tensor of shape (num_samples,).
        """


        # Calculate power of speech and noise
        speech_power = speech.pow(2).mean().clamp(min=1e-10)
        noise_power = noise.pow(2).mean().clamp(min=1e-10)

        # Calculate required noise power for desired SNR
        snr_linear = 10 ** (snr_db / 10)
        required_noise_power = speech_power / snr_linear
        
        # Scale noise to achieve desired SNR
        scaling_factor = (required_noise_power / noise_power).sqrt()
        noisy_speech = speech + noise * scaling_factor
        return noisy_speech

    def __len__(self):
        return len(self.train_speech)
    
    def __getitem__(self, idx, split='train'):
        # Load speech, noise, and RIR files
        if split == 'train':
            speech_path = self.train_speech[idx % len(self.train_speech)]
            noise_path = self.train_noise[idx % len(self.train_noise)]
            rir_path = self.train_rir[idx % len(self.train_rir)]
        else:
            speech_path = self.val_speech[idx % len(self.val_speech)]
            noise_path = self.val_noise[idx % len(self.val_noise)]
            rir_path = self.val_rir[idx % len(self.val_rir)]
        
        speech, sr = torchaudio.load(speech_path)
        noise, _ = torchaudio.load(noise_path)
        rir, _ = torchaudio.load(rir_path)
        
        if sr != self.sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sr)
            speech = resampler(speech)
            noise = resampler(noise)
            rir = resampler(rir)
            sr = 16_000
        
        if speech.dim() > 1:
            speech = speech[0, :]
        if noise.dim() > 1:
            noise = noise[0, :]
        if rir.dim() > 1:
            rir = rir[0, :]
        
        #Make sure speech and noise are segment_length long
        segment_samples = int(self.segment_length * self.sr)
        if speech.shape[0] > segment_samples:
            start = random.randint(0, speech.shape[0] - segment_samples)
            speech = speech[start:start + segment_samples]
        else:
            speech = F.pad(speech, (0, segment_samples - speech.shape[0]))
        
        if noise.shape[0] > segment_samples:
            start = random.randint(0, noise.shape[0] - segment_samples)
            noise = noise[start:start + segment_samples]
        else:
            noise = F.pad(noise, (0, segment_samples - noise.shape[0]))

        #Apply early reverb to speech and get clean speech
        early_rir = self.get_early_reverbs(rir, self.sr)
        clean_speech = self.apply_reverb(speech, early_rir)

        #Apply full reverb to speech
        reverberated_speech = self.apply_reverb(speech, rir)

        #Add noise with a probability of 0.8
        if random.random() < 0.8:
            snr = random.uniform(self.global_snr_range[0], self.global_snr_range[1])
            noisy_speech = self.add_noise(reverberated_speech, noise, snr)
        else:
            noisy_speech = reverberated_speech
        
        return noisy_speech, clean_speech


class ORDataModule(pl.LightningDataModule):
    def __init__(self, speech_list, noise_list, rir_list, batch_size=8, num_workers=4, pin_memory=True, **kwargs):
        super(ORDataModule, self).__init__()
        self.speech_list = speech_list
        self.noise_list = noise_list
        self.rir_list = rir_list
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataset_kwargs = kwargs

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = ORDataset(self.speech_list, self.noise_list, self.rir_list, split='train', **self.dataset_kwargs)
        self.val_dataset = ORDataset(self.speech_list, self.noise_list, self.rir_list, split='val', **self.dataset_kwargs)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)


        


if __name__ == "__main__":
    
    ds = ORDataset(
        speech_list="utils/segments_speech8s.txt",
        noise_list="utils/segments_noise8s.txt",
        rir_list="utils/rirs.txt",
        batch_size=16,
        num_workers=0,
        sr=16000,
        segment_length=8.0,
        global_snr=(-5, 10),
        add_noise_prob=0.8,
        early_reverb_sec=0.05,
        )
    noisy, clean = ds[0]
    
    dm = ORDataModule(
        speech_list="utils/segments_speech8s.txt",
        noise_list="utils/segments_noise8s.txt",
        rir_list="utils/rirs.txt",
        batch_size=16,
        num_workers=0,
        sr=16000,
        segment_length=8.0,
        global_snr=(-5, 10),
        add_noise_prob=0.8,
        early_reverb_sec=0.05,
        )
    
        