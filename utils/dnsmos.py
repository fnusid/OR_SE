import numpy as np
import librosa
import onnxruntime as ort

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01
LEN_SAMPLES = int(SAMPLING_RATE * INPUT_LENGTH)

class DNSMOSMetric:
    """
    File-free, batched DNSMOS scorer for use in training loops.
    Accepts torch.Tensor [B,T] or numpy [B,T].
    """

    def __init__(self, primary_model_path: str, p808_model_path: str,
                 personalized: bool = False, providers=None,
                 intra_threads: int | None = None, inter_threads: int | None = None):
        so = ort.SessionOptions()
        if intra_threads is not None: so.intra_op_num_threads = int(intra_threads)
        if inter_threads is not None: so.inter_op_num_threads = int(inter_threads)
        self.primary_sess = ort.InferenceSession(primary_model_path, so, providers=providers or ort.get_available_providers())
        self.p808_sess    = ort.InferenceSession(p808_model_path,    so, providers=providers or ort.get_available_providers())
        self.personalized = personalized

        # Precompile polyfits
        if personalized:
            self.p_ovr = np.poly1d([-0.00533021,  0.005101,    1.18058466, -0.11236046])
            self.p_sig = np.poly1d([-0.01019296,  0.02751166,  1.19576786, -0.24348726])
            self.p_bak = np.poly1d([-0.04976499,  0.44276479, -0.1644611,   0.96883132])
        else:
            self.p_ovr = np.poly1d([-0.06766283,  1.11546468,  0.04602535])
            self.p_sig = np.poly1d([-0.08397278,  1.22083953,  0.0052439 ])
            self.p_bak = np.poly1d([-0.13166888,  1.60915514, -0.39604546])

    # ---------- helpers ----------
    @staticmethod
    def _to_numpy_batch(wavs):
        try:
            import torch
            if isinstance(wavs, torch.Tensor):
                return wavs.detach().to("cpu", copy=False).float().numpy()
        except Exception:
            pass
        arr = np.asarray(wavs)
        assert arr.ndim == 2, "wavs must be [B, T]"
        return arr.astype(np.float32)

    @staticmethod
    def _resample_pad_trim(x: np.ndarray, sr: int) -> np.ndarray:
        """x: [T]; resample to 16k, then pad/trim to LEN_SAMPLES (>=)."""
        if sr != SAMPLING_RATE:
            x = librosa.resample(x, orig_sr=sr, target_sr=SAMPLING_RATE)
        if len(x) < LEN_SAMPLES:
            reps = int(np.ceil(LEN_SAMPLES / max(1, len(x))))
            x = np.tile(x, reps)
        return x[:LEN_SAMPLES].astype(np.float32)

    @staticmethod
    def _melspec_120(x16k: np.ndarray) -> np.ndarray:
        """Mel spec for P808 input. Returns [T, 120] in float32, normalized like original code."""
        mel = librosa.feature.melspectrogram(
            y=x16k, sr=SAMPLING_RATE, n_fft=321, hop_length=160, n_mels=120
        )
        mel = (librosa.power_to_db(mel, ref=np.max) + 40.0) / 40.0
        return mel.T.astype(np.float32)

    # ---------- public API ----------
    def score_torch_batch(self, wavs, sr: int):
        """
        wavs: torch.Tensor or np.ndarray of shape [B, T] at sample rate sr.
        Returns dict of numpy arrays (length B):
            SIG_raw, BAK_raw, OVRL_raw, SIG, BAK, OVRL, P808_MOS
        """
        batch = self._to_numpy_batch(wavs)     # [B, T]
        B = batch.shape[0]

        # Resample/pad/trim each item to exactly LEN_SAMPLES @ 16k
        proc = [self._resample_pad_trim(batch[i], sr) for i in range(B)]

        # --- Primary model (waveform) batched ---
        wave_inp = np.stack(proc, axis=0).astype(np.float32)    # [B, N]
        primary_out = self.primary_sess.run(None, {'input_1': wave_inp})[0]  # [B, 3]
        sig_raw = primary_out[:, 0]; bak_raw = primary_out[:, 1]; ovr_raw = primary_out[:, 2]

        # --- P808 (mel) batched (pad to max T across batch) ---
        # Use last 160-sample crop as in your code (seg[:-160])
        mels = [self._melspec_120(p[:-160]) for p in proc]      # each [Ti, 120]
        maxT = max(m.shape[0] for m in mels); n_mels = mels[0].shape[1]
        mel_pad = np.zeros((B, maxT, n_mels), dtype=np.float32)
        for i, m in enumerate(mels):
            mel_pad[i, :m.shape[0]] = m
        p808 = self.p808_sess.run(None, {'input_1': mel_pad})[0][:, 0]    # [B]

        # --- Polyfit calibration ---
        SIG = self.p_sig(sig_raw)
        BAK = self.p_bak(bak_raw)
        OVRL = self.p_ovr(ovr_raw)

        return {
            "SIG_raw": sig_raw.astype(np.float32),
            "BAK_raw": bak_raw.astype(np.float32),
            "OVRL_raw": ovr_raw.astype(np.float32),
            "SIG": SIG.astype(np.float32),
            "BAK": BAK.astype(np.float32),
            "OVRL": OVRL.astype(np.float32),
            "P808_MOS": p808.astype(np.float32),
        }


if __name__ == "__main__":
    scorer = DNSMOSMetric(
    primary_model_path="DNSMOS/sig_bak_ovr.onnx",   # or "pDNSMOS/sig_bak_ovr.onnx"
    p808_model_path="DNSMOS/model_v8.onnx",
    personalized=False,            # True if using personalized model
    providers=None                 # e.g. ["CUDAExecutionProvider","CPUExecutionProvider"]
    )
    # breakpoint()
    import torch
    preds = torch.randn(4, 16000*10)  # 4 random 10-second clips
    results = scorer.score_torch_batch(preds, sr=16000)
    breakpoint()
    '''
            self.log_dict({
            "dnsmos/ovrl": float(scores["OVRL"].mean()),
            "dnsmos/sig":  float(scores["SIG"].mean()),
            "dnsmos/bak":  float(scores["BAK"].mean()),
            "dnsmos/p808": float(scores["P808_MOS"].mean()),
        }, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
    '''