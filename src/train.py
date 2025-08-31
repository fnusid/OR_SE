import torch
from src.model import ORSEModel
from src.dataset import ORDataModule
import src.config as config
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
import torch.multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)  # avoid fork deadlocks on SLURM
except RuntimeError:
    pass

# Optional: use soundfile backend; avoids some torchaudio/sox quirks
import torchaudio
try:
    torchaudio.set_audio_backend("soundfile")
except Exception:
    pass

"""
If you want to run thsi code, just follow the command below
python train.py

Note; If you want to format the code, use the command below

pip install black
black .
"""


if __name__ == "__main__":
    logger = WandbLogger(project=config.project, name=config.model_name)

    model = ORSEModel(
        frame_len = config.frame_len,
        hop_len = config.hop_len,
        fft_len = config.fft_len,
        win = config.win,
        length = config.length,
        enc_filters = config.enc_filters,
        enc_kernels = config.enc_kernels,
        enc_strides = config.enc_strides,
        in_channels = config.in_channels, # real and imag, or mag and phase
        num_bottleneck_layers = config.num_bottleneck_layers,
        loss_fn = config.loss_fn,
        loss_weights = config.loss_weights,
        lr = config.lr,
        alpha=config.alpha, 
        metric=config.metric,
    )
    # Load the data modules
    dm = ORDataModule(
        speech_list=config.speech_list,
        noise_list=config.noise_list,
        rir_list=config.rir_list,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sr=config.sr,
        segment_length=config.segment_length,
        global_snr=config.global_snr,
        add_noise_prob=config.add_noise_prob,
        early_reverb_sec=config.early_reverb_sec,
        peak_normalize = config.peak_normalization
    )

    # Train the model using pl trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        detect_anomaly=True,
        gradient_clip_val=1.0,   
        check_val_every_n_epoch=config.check_val_every_n_epoch,
        log_every_n_steps=config.log_every_n_steps,
        enable_checkpointing=config.enable_checkpointing,
        accelerator=config.accelerator,
        devices=config.devices,

        callbacks=[
            EarlyStopping(monitor='val_loss', patience=10, mode='min'),
            ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=3, filename='best-checkpoint-{epoch:02d}-{val_loss:.2f}')
        ],
        logger=logger,
    )
    trainer.fit(model, dm, ckpt_path=config.ckpt_path)
    trainer.validate(model, dm)