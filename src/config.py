#Compute related
accelerator="gpu"
devices=4


#Dataset params
speech_list="utils/segments_speech8s.txt"
noise_list="utils/segments_noise8s.txt"
rir_list="utils/rirs.txt"
batch_size=64
num_workers=2
sr=16_000
segment_length=8.0
global_snr=(-5, 10)
add_noise_prob=0.8
early_reverb_sec=0.05

#Model params
frame_len = 512
hop_len = 256
fft_len = 512
win = 'sine'
length = 8
enc_filters = [16,32,64,64,128,128,256]
enc_kernels = [(5,1), (3,1), (3,1), (3,1), (3,1), (3,1), (3, 1)] #[F,T] keeping it causal
enc_strides = [(2,1), (2,1), (2,1), (2,1), (2,1), (2,1), (2, 1)]
in_channels = 2 # real and imag, or mag and phase
num_bottleneck_layers = 2
loss_fn = ["mse", "complex_spectral"]
loss_weights = [0.5, 0.5]
lr = 1e-3
alpha=0.5 
metric='DNSMOS'
world_size = 4

#Trainer params
max_epochs=400
check_val_every_n_epoch=5
log_every_n_steps=10
enable_checkpointing=True
ckpt_path="/scratch/profdj_root/profdj0/sidcs/codebase/or_se/or_speech_enhancement/4nb6krjq/checkpoints/best-checkpoint-epoch=84-val_loss=0.23.ckpt"

#wandb params
project="or_speech_enhancement"
model_name="baseline"