import os
import PIL
import gc
from fastai.torch_core import requires_grad, children
from fastai.torch_core import F, torch, hook_outputs, nn
from fastai.core import parallel
from functools import partial
from fastai.datasets import untar_data, URLs
from fastai.basic_data import DatasetType
from torchvision.models import vgg16_bn
from fastai.callbacks import LossMetrics
from fastai.vision import ImageList, ImageImageList, resize_to, models
from fastai.vision import get_transforms
from fastai.vision.data import imagenet_stats
from fastai.vision.learner import unet_learner
from fastai.layers import NormType
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


path = untar_data(URLs.PETS)
path_hr = path/'images'
path_lr = path/'small-96'
path_mr = path/'small-256'

il = ImageList.from_folder(path_hr)


def resize_one(fn, i, path, size):
    dest = path/fn.relative_to(path_hr)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = PIL.Image.open(fn)
    targ_sz = resize_to(img, size, use_min=True)
    img = img.resize(targ_sz, resample=PIL.Image.BILINEAR).convert('RGB')
    img.save(dest, quality=60)


# create smaller image sets the first time this nb is run
sets = [(path_lr, 96), (path_mr, 256)]
for p, size in sets:
    if not p.exists():
        print(f"resizing to {size} into {p}")
        parallel(partial(resize_one, path=p, size=size), il.items)

bs, size = 32, 128
arch = models.resnet34

src = ImageImageList.from_folder(path_lr).split_by_rand_pct(0.1, seed=42)


def get_data(bs, size):
    data = (src.label_from_func(lambda x: path_hr/x.name)
            .transform(get_transforms(max_zoom=2.), size=size, tfm_y=True)
            .databunch(bs=bs).normalize(imagenet_stats, do_y=True))

    data.c = 3
    return data


data = get_data(bs, size)


data.show_batch(ds_type=DatasetType.Valid, rows=2, figsize=(9, 9))

t = data.valid_ds[0][1].data
t = torch.stack([t, t])


def gram_matrix(x):
    n, c, h, w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1, 2))/(c*h*w)


gram_matrix(t)

base_loss = F.l1_loss

vgg_m = vgg16_bn(True).features.cuda().eval()
requires_grad(vgg_m, False)

blocks = [i-1 for i,
          o in enumerate(children(vgg_m)) if isinstance(o, nn.MaxPool2d)]
blocks, [vgg_m[i] for i in blocks]


class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts):
        super().__init__()
        self.m_feat = m_feat
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel', ] + [f'feat_{i}' for i in range(
            len(layer_ids))] + [f'gram_{i}' for i in range(len(layer_ids))]

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]

    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [base_loss(input, target)]
        self.feat_losses += [base_loss(f_in, f_out)*w for f_in,
                             f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [base_loss(gram_matrix(f_in),
                                       gram_matrix(f_out)) *
                             w**2 * 5e3 for f_in, f_out, w in
                             zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)

    def __del__(self): self.hooks.remove()


feat_loss = FeatureLoss(vgg_m, blocks[2:5], [5, 15, 2])

wd = 1e-3
learn = unet_learner(data, arch, wd=wd, loss_func=feat_loss,
                     callback_fns=LossMetrics,
                     blur=True, norm_type=NormType.Weight)
gc.collect()

lr = 1e-3


def do_fit(save_name, lrs=slice(lr), pct_start=0.9):
    learn.fit_one_cycle(10, lrs, pct_start=pct_start)
    learn.save(save_name)
    learn.show_results(rows=1, imgsize=5)


wandbCallbacks = True

if wandbCallbacks:
    import wandb
    from wandb.fastai import WandbCallback
    from datetime import datetime

    config = {"batch_size": bs,
              "learning_rate": lr,
              "weight_decay": wd
              }
    wandb.init(project='SuperRes', config=config,
               id="gen_resnet34" + datetime.now().strftime('_%m-%d_%H:%M'))

    learn.callback_fns.append(partial(WandbCallback, input_type='images'))

do_fit('1a', slice(lr*10))

learn.unfreeze()

do_fit('1b', slice(1e-5, lr))

data = get_data(12, size*2)

learn.data = data
learn.freeze()
gc.collect()

learn.load('1b')

do_fit('2a')

learn.unfreeze()

do_fit('2b', slice(1e-6, 1e-4), pct_start=0.3)

learn = None
gc.collect()
