import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from typing import Optional, Any

class ORDataModule(nn.Module):
    def __init__(self, speech_list, noise_list, rir_list, batch_size=8, num_workers=4, pin_memory=True):
        super(ORDataModule, self).__init__()
        self.speech_list = speech_list
        self.noise_list = noise_list
        self.rir_list = rir_list
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory