import torch
import torch.nn as nn
import sys
sys.path.append('/scratch/profdj_root/profdj0/shared_data/DNS-Challenge/DNSMOS/')
from dnsmos import DNSMOSMetric


class MetricWrapper(nn.Module):
    def __init__(self, metric):
        super(MetricWrapper, self).__init__()
        if metric == 'DNSMOS':
            self.metric = DNSMOSMetric(
                primary_model_path="DNSMOS/sig_bak_ovr.onnx",   # or "pDNSMOS/sig_bak_ovr.onnx"
                p808_model_path="DNSMOS/model_v8.onnx",
                personalized=False,            # True if using personalized model
                providers=None                 # e.g. ["CUDAExecutionProvider","CPUExecutionProvider"]
            )
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
    def forward(self, wavs, sr):
        metric_scores = self.metric.score_torch_batch(wavs, sr)
        SIG = metric_scores['SIG'].mean()
        BAK = metric_scores['BAK'].mean()
        OVRL = metric_scores['OVRL'].mean()

        return {'SIG': SIG, 'BAK': BAK, 'OVRL': OVRL}