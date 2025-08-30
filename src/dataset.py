import os
import random
from typing import Optional, Any, List, Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchaudio
import pytorch_lightning as pl


# ---------------------------
# DataLoader helpers
# ---------------------------

def _worker_init_fn(worker_id: int):
    # keep worker processes from oversubscribing CPU threads
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)

    # deterministic-ish seeds per worker
    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


def collate_pair(batch: List[Dict[str, torch.Tensor]]):
    # batch fixed-length 1D tensors as (B, T)
    noisy = torch.stack([b["noisy"] for b in batch], dim=0)
    clean = torch.stack([b["clean"] for b in batch], dim=0)
    return noisy, clean


# ---------------------------
# Dataset
# ---------------------------

class ORDataset(Dataset):
    """
    Expects lists of filepaths for speech/noise/rir. Produces dicts:
        {"noisy": (T,), "clean": (T,)}
    """

    def __init__(
        self,
        speech: List[str],
        noise: List[str],
        rir: List[str],
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        self.speech = speech
        self.noise = noise
        self.rir = rir

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # config
        self.sr = int(kwargs.get("sr", 16000))
        self.segment_length = float(kwargs.get("segment_length", 8.0))
        self.add_noise_prob = float(kwargs.get("add_noise_prob", 0.8))
        self.early_reverb_sec = float(kwargs.get("early_reverb_sec", 0.05))
        self.global_snr_range = tuple(kwargs.get("global_snr", (-5, 10)))  # dB
        # optional cap on RIR length for speed (seconds). None = no cap
        self.max_rir_seconds = kwargs.get("max_rir_seconds", 0.5)

        # per-worker lightweight caches
        self._rir_cache: Dict[str, torch.Tensor] = {}
        self._resampler_cache: Dict[tuple, torchaudio.transforms.Resample] = {}

    # -------- I/O utils --------

    def _get_resampler(self, src_sr: int) -> Optional[torchaudio.transforms.Resample]:
        if src_sr == self.sr:
            return None
        key = (src_sr, self.sr)
        rs = self._resampler_cache.get(key)
        if rs is None:
            # torchscript-friendly, fast resampler
            rs = torchaudio.transforms.Resample(src_sr, self.sr)
            self._resampler_cache[key] = rs
        return rs

    def _load_wav_mono(self, path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(path)  # (C, T)
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav[0:1]  # take first channel
        wav = wav.squeeze(0).to(torch.float32)
        rs = self._get_resampler(sr)
        if rs is not None:
            wav = rs(wav.unsqueeze(0)).squeeze(0)
        return wav

    def _load_rir(self, path: str) -> torch.Tensor:
        if path in self._rir_cache:
            return self._rir_cache[path]
        rir, sr = torchaudio.load(path)  # (C, T)
        rir = rir[0].to(torch.float32) if rir.dim() == 2 else rir.to(torch.float32)
        rs = self._get_resampler(sr)
        if rs is not None:
            rir = rs(rir.unsqueeze(0)).squeeze(0)
        # optional cap for speed (e.g., 0.5 s @ 16k -> 8000 taps)
        if self.max_rir_seconds and self.max_rir_seconds > 0:
            max_taps = int(self.max_rir_seconds * self.sr)
            rir = rir[:max_taps]
        self._rir_cache[path] = rir
        return rir

    # -------- DSP helpers --------

    @staticmethod
    def _fft_convolve_same_len(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Fast convolution via FFT, returning first len(x) samples.
        x, h: (T,)
        """
        T = x.numel()
        n = T + h.numel() - 1
        nfft = 1 << (n - 1).bit_length()  # next power of two

        X = torch.fft.rfft(x, n=nfft)
        H = torch.fft.rfft(h, n=nfft)
        y = torch.fft.irfft(X * H, n=nfft)[:T]
        return y

    @staticmethod
    def _early_window(rir: torch.Tensor, sr: int, duration: float, eps: float = 1e-8) -> torch.Tensor:
        """
        Extract 'early' RIR (starts at first non-negligible sample) and
        pad back to original length so indexing stays simple.
        """
        r = rir.flatten()
        nz = torch.nonzero(r.abs() > eps, as_tuple=False)
        offset = int(nz.min()) if nz.numel() else 0
        shifted = r[offset:]
        early_len = min(int(sr * duration), shifted.numel())
        early = shifted[:early_len]
        return F.pad(early, (0, r.numel() - early.numel()))

    @staticmethod
    def _add_noise_at_snr(speech: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
        sp = speech.pow(2).mean().clamp_min(1e-10)
        npow = noise.pow(2).mean().clamp_min(1e-10)
        snr_lin = 10.0 ** (snr_db / 10.0)
        req_np = sp / snr_lin
        scale = (req_np / npow).sqrt()
        return speech + noise * scale

    # -------- Dataset API --------

    def __len__(self):
        return len(self.speech)

    def __getitem__(self, idx: int):
        sp_path = self.speech[idx % len(self.speech)]
        nz_path = self.noise[idx % len(self.noise)]
        rr_path = self.rir[idx % len(self.rir)]

        speech = self._load_wav_mono(sp_path)
        noise = self._load_wav_mono(nz_path)
        rir = self._load_rir(rr_path)

        # crop/pad to segment length
        seg_T = int(self.segment_length * self.sr)

        if speech.numel() > seg_T:
            s0 = random.randint(0, speech.numel() - seg_T)
            speech = speech[s0 : s0 + seg_T]
        else:
            speech = F.pad(speech, (0, seg_T - speech.numel()))

        if noise.numel() > seg_T:
            n0 = random.randint(0, noise.numel() - seg_T)
            noise = noise[n0 : n0 + seg_T]
        else:
            noise = F.pad(noise, (0, seg_T - noise.numel()))

        # early (target) and full reverb via FFT conv
        early_rir = self._early_window(rir, self.sr, self.early_reverb_sec)
        clean = self._fft_convolve_same_len(speech, early_rir)
        reverb = self._fft_convolve_same_len(speech, rir)

        if random.random() < self.add_noise_prob:
            snr = random.uniform(self.global_snr_range[0], self.global_snr_range[1])
            noisy = self._add_noise_at_snr(reverb, noise, snr)
        else:
            noisy = reverb

        return {"noisy": noisy, "clean": clean}


# ---------------------------
# DataModule
# ---------------------------

class ORDataModule(pl.LightningDataModule):
    def __init__(
        self,
        speech_list: str,
        noise_list: str,
        rir_list: str,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        self.speech_list = speech_list
        self.noise_list = noise_list
        self.rir_list = rir_list

        self.batch_size = batch_size
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.dataset_kwargs = kwargs

        # read filelists
        with open(self.speech_list) as f:
            self.speech_files = [ln.strip() for ln in f if ln.strip()]
        with open(self.noise_list) as f:
            self.noise_files = [ln.strip() for ln in f if ln.strip()]
        with open(self.rir_list) as f:
            self.rir_files = [ln.strip() for ln in f if ln.strip()]

        # shuffle once before split
        random.shuffle(self.speech_files)
        random.shuffle(self.noise_files)
        random.shuffle(self.rir_files)

        # split 80/20
        ns, nn, nr = map(len, (self.speech_files, self.noise_files, self.rir_files))
        s_cut, n_cut, r_cut = int(0.8 * ns), int(0.8 * nn), int(0.8 * nr)

        self.train_speech = self.speech_files[:s_cut]
        self.val_speech = self.speech_files[s_cut:]

        self.train_noise = self.noise_files[:n_cut]
        self.val_noise = self.noise_files[n_cut:]

        self.train_rir = self.rir_files[:r_cut]
        self.val_rir = self.rir_files[r_cut:]

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = ORDataset(
            self.train_speech, self.train_noise, self.train_rir, **self.dataset_kwargs
        )
        self.val_dataset = ORDataset(
            self.val_speech, self.val_noise, self.val_rir, **self.dataset_kwargs
        )

    def train_dataloader(self):
        use_workers = self.num_workers > 0
        # prefetch_factor must be omitted when num_workers == 0
        kwargs = dict(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=use_workers,
            drop_last=True,
            collate_fn=collate_pair,
            timeout=0,
            worker_init_fn=_worker_init_fn,
        )
        if use_workers:
            kwargs["prefetch_factor"] = 2  # good general default
        return DataLoader(**kwargs)

    def val_dataloader(self):
        # keep workers at 0 during sanity check to avoid pickling issues in some envs
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=self.pin_memory,
            persistent_workers=False,
            drop_last=False,
            collate_fn=collate_pair,
            timeout=0,
            worker_init_fn=_worker_init_fn,
        )


# ---------------------------
# Quick local smoke test
# ---------------------------
if __name__ == "__main__":
    # expects lists of file paths; change these if you want to run locally
    breakpoint()
    dm = ORDataModule(
        speech_list="utils/segments_speech8s.txt",
        noise_list="utils/segments_noise8s.txt",
        rir_list="utils/rirs.txt",
        batch_size=8,
        num_workers=4,
        pin_memory=True,
        sr=16000,
        segment_length=8.0,
        global_snr=(-5, 10),
        add_noise_prob=0.8,
        early_reverb_sec=0.05,
        max_rir_seconds=0.5,  # <- speed cap; set None to disable
    )
    dm.setup()
    dl = dm.train_dataloader()
    x, y = next(iter(dl))
    print(x.shape, y.shape)  # (B, T) (B, T)
