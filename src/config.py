#Compute related
accelerator="gpu"
devices=1


#Dataset params
speech_list="utils/segments_speech8s.txt"
noise_list="utils/segments_noise8s.txt"
rir_list="utils/rirs.txt"
batch_size=16
num_workers=10
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
length = 1
enc_filters = [16,32,64,64,128,256]
enc_kernels = [(5,1), (3,1), (3,1), (3,1), (3,1), (3,1)] #[F,T] keeping it causal
enc_strides = [(2,1), (2,1), (2,1), (2,1), (2,1), (2,1)]
in_channels = 2 # real and imag, or mag and phase
num_bottleneck_layers = 2
loss_fn = ['mse', 'complex_spectral']
loss_weights = [0.5, 0.5]
lr = 1e-3
alpha=0.5 
metric='DNSMOS'

#Trainer params
max_epochs=400
check_val_every_n_epoch=5
log_every_n_steps=10
enable_checkpointing=True
ckpt_path='checkpoints/'