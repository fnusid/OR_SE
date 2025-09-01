import torch
import torch.nn as nn
import sys
import numpy as np

from utils.dnsmos import DNSMOSMetric
from pesq import pesq_batch


class MetricWrapper(nn.Module):
    def __init__(self, metrics):
        super(MetricWrapper, self).__init__()
        self.metrics_names = metrics
        self.metrics_fn_list = []
        for metric in metrics:
            if metric == 'DNSMOS':
                self.metric = DNSMOSMetric(
                    primary_model_path="utils/DNSMOS/sig_bak_ovr.onnx",   # or "pDNSMOS/sig_bak_ovr.onnx"
                    p808_model_path="utils/DNSMOS/model_v8.onnx",
                    personalized=False,            # True if using personalized model
                    providers=None                 # e.g. ["CUDAExecutionProvider","CPUExecutionProvider"]
                )
                self.metrics_fn_list.append(self.metric)
            elif metric == 'PESQ':
                self.metric = pesq_batch
                self.metrics_fn_list.append(self.metric)

            else:
                raise ValueError(f"Unsupported metric: {metric}")
            
    def forward(self, wavs,sr, gts=None):
        metric_values = {}
        for i, metric_fn in enumerate(self.metrics_fn_list):
            if self.metrics_names[i] == 'PESQ':
                metric_scores = metric_fn(sr, gts.detach().numpy(), wavs.detach().numpy(), mode='wb')
                metric_values['PESQ'] = np.mean(metric_scores).item()
            elif self.metrics_names[i]=='DNSMOS':
                metric_scores = metric_fn.score_torch_batch(wavs, sr)
                SIG = metric_scores['SIG'].mean()
                BAK = metric_scores['BAK'].mean()
                OVRL = metric_scores['OVRL'].mean()
                metric_values['SIG'] = SIG
                metric_values['BAK'] = BAK
                metric_values['OVRL'] = OVRL 
            else:
                raise ValueError(f"Unsupported metric: {self.metrics_names[i]}")

        return metric_values