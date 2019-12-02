from fastai.torch_core import add_metrics
from fastai.callback import Callback
from brisque import BRISQUE
from skvideo.measure.niqe import niqe
from .ssim import ssim

import perceptual_similarity as lpips


class SSIM_Metric(Callback):
    def __init__(self):
        super().__init__()
        self.name = "ssim"

    def on_epoch_begin(self, **kwargs):
        self.values = []

    def on_batch_end(self, last_output, last_target, **kwargs):
        self.values.append(ssim(last_output, last_target))

    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics, sum(self.values) / len(self.values))


class LPIPS_Metric(Callback):
    def __init__(self):
        super().__init__()
        self.name = "lpips"
        self.model = lpips.PerceptualLoss(
            model='net-lin', net='alex', use_gpu=True, gpu_ids=[0])

    def on_epoch_begin(self, **kwargs):
        self.values = []

    def on_batch_end(self, last_output, last_target, **kwargs):
        lpips_value = self.model.forward(last_output, last_target)
        self.values.append(lpips_value.mean())

    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics, sum(self.values) / len(self.values))


class BRISQUE_Metric(Callback):
    def __init__(self):
        super().__init__()
        self.name = "brisque"
        self.brisque = BRISQUE()

    def on_epoch_begin(self, **kwargs):
        self.values = []

    def on_batch_end(self, last_output, last_target, **kwargs):
        for img in last_output:
            score = self.brisque.get_score(img.permute(1, 2, 0).cpu().numpy())
            self.values.append(score)

    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics, sum(self.values) / len(self.values))


class NIQE_Metric(Callback):
    def __init__(self):
        super().__init__()
        self.name = "niqe"

    def on_epoch_begin(self, **kwargs):
        self.values = []

    def on_batch_end(self, last_output, last_target, **kwargs):
        for img in last_output:
            score = niqe(img[0].cpu().numpy())
            self.values.append(score)

    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics, sum(self.values) / len(self.values))
