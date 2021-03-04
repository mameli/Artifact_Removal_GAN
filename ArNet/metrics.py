from fastai.torch_core import add_metrics
from fastai.callback import Callback
from brisque import BRISQUE
from skvideo.measure.niqe import niqe
from .ssim import ssim

import lpips


class SSIM_Metric_gen(Callback):
    def __init__(self):
        super().__init__()
        self.name = "ssim_gen"

    def on_epoch_begin(self, **kwargs):
        self.values = []

    def on_batch_end(self, last_output, last_target, **kwargs):
        self.values.append(1 - ssim(last_output, last_target))

    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics, sum(self.values) / len(self.values))


class SSIM_Metric_input(Callback):
    def __init__(self):
        super().__init__()
        self.name = "ssim_in"
        self.final_score = 0.

    def on_epoch_begin(self, **kwargs):
        self.values = []

    def on_batch_begin(self, last_input, last_target, **kwargs):
        self.values.append(1 - ssim(last_input, last_target))

    def on_epoch_end(self, last_metrics, **kwargs):
        self.final_score = sum(self.values) / len(self.values)
        return add_metrics(last_metrics, self.final_score)


class LPIPS_Metric_gen(Callback):
    def __init__(self):
        super().__init__()
        self.name = "lpips_gen"
        self.model = lpips.LPIPS(net='alex', spatial=True)

    def on_epoch_begin(self, **kwargs):
        self.values = []

    def on_batch_end(self, last_output, last_target, **kwargs):
        lpips_value = self.model.forward(last_output, last_target)
        self.values.append(lpips_value.mean())

    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics, sum(self.values) / len(self.values))


class LPIPS_Metric_input(Callback):
    def __init__(self):
        super().__init__()
        self.name = "lpips_in"
        self.final_score = 0.
        self.model = lpips.LPIPS(net='alex', spatial=True)

    def on_epoch_begin(self, **kwargs):
        self.values = []

    def on_batch_begin(self, last_input, last_target, **kwargs):
        lpips_value = self.model.forward(last_input, last_target)
        self.values.append(lpips_value.mean())

    def on_epoch_end(self, last_metrics, **kwargs):
        self.final_score = sum(self.values) / len(self.values)
        return add_metrics(last_metrics, self.final_score)


class BRISQUE_Metric_gen(Callback):
    def __init__(self):
        super().__init__()
        self.name = "brisque_gen"
        self.brisque = BRISQUE()

    def on_epoch_begin(self, **kwargs):
        self.values = []

    def on_batch_end(self, last_output, last_target, **kwargs):
        for img in last_output:
            score = self.brisque.get_score(img.permute(1, 2, 0).cpu().numpy())
            self.values.append(score)

    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics, sum(self.values) / len(self.values))


class BRISQUE_Metric_input(Callback):
    def __init__(self):
        super().__init__()
        self.name = "brisque_in"
        self.brisque = BRISQUE()
        self.final_score = 0.

    def on_epoch_begin(self, **kwargs):
        self.values = []

    def on_batch_begin(self, last_input, last_target, **kwargs):
        for img in last_input:
            score = self.brisque.get_score(
                img.permute(1, 2, 0).cpu().numpy())
            self.values.append(score)

    def on_epoch_end(self, last_metrics, **kwargs):
        self.final_score = sum(self.values) / len(self.values)
        return add_metrics(last_metrics, self.final_score)


class BRISQUE_Metric_target(Callback):
    def __init__(self):
        super().__init__()
        self.name = "brisque_tar"
        self.brisque = BRISQUE()
        self.final_score = 0.

    def on_epoch_begin(self, **kwargs):
        self.values = []

    def on_batch_begin(self, last_input, last_target, **kwargs):
        for img in last_target:
            score = self.brisque.get_score(
                img.permute(1, 2, 0).cpu().numpy())
            self.values.append(score)

    def on_epoch_end(self, last_metrics, **kwargs):
        self.final_score = sum(self.values) / len(self.values)
        return add_metrics(last_metrics, self.final_score)


class NIQE_Metric_gen(Callback):
    def __init__(self):
        super().__init__()
        self.name = "niqe_gen"

    def on_epoch_begin(self, **kwargs):
        self.values = []

    def on_batch_end(self, last_output, last_target, **kwargs):
        for img in last_output:
            score = niqe(img[0].cpu().numpy())
            self.values.append(score)

    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics, sum(self.values) / len(self.values))


class NIQE_Metric_input(Callback):
    def __init__(self):
        super().__init__()
        self.name = "niqe_in"
        self.final_score = 0.

    def on_epoch_begin(self, **kwargs):
        self.values = []

    def on_batch_begin(self, last_input, last_target, **kwargs):
        for img in last_input:
            score = niqe(img[0].cpu().numpy())
            self.values.append(score)

    def on_epoch_end(self, last_metrics, **kwargs):
        self.final_score = sum(self.values) / len(self.values)
        return add_metrics(last_metrics, self.final_score)


class NIQE_Metric_target(Callback):
    def __init__(self):
        super().__init__()
        self.name = "niqe_tar"
        self.final_score = 0.

    def on_epoch_begin(self, **kwargs):
        self.values = []

    def on_batch_begin(self, last_input, last_target, **kwargs):
        for img in last_target:
            score = niqe(img[0].cpu().numpy())
            self.values.append(score)

    def on_epoch_end(self, last_metrics, **kwargs):
        self.final_score = sum(self.values) / len(self.values)
        return add_metrics(last_metrics, self.final_score)
