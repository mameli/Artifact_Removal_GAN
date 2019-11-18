import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

from fastai import *
from fastai.vision import *
from superRes.generators import *
from superRes.critics import *
from superRes.dataset import *
from superRes.loss import *
from superRes.save import *
from superRes.fid_loss import *
from superRes.ssim import *
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageFile
from pathlib import Path

import geffnet # efficient/ mobile net
from datetime import datetime

import geffnet # efficient/ mobile net

def get_data(bs:int, sz:int, keep_pct:float):
    return get_databunch(sz=sz, bs=bs, crappy_path=path_lowRes, 
                         good_path=path_fullRes, 
                         random_seed=None, keep_pct=keep_pct)

def det_DIV2k_data(bs:int, sz:int):
    lowResSuffix = 'x4m'
    src = ImageImageList.from_folder(path_lowRes).split_by_idxs(train_idx=list(range(0,800)), valid_idx=list(range(800,900)))

    data = (src.label_from_func(lambda x: path_fullRes/(x.name).replace(lowResSuffix, '')).transform(
            get_transforms(
                max_zoom=2.
            ),
            size=sz,
            tfm_y=True,
        ).databunch(bs=bs, num_workers=8, no_check=True).normalize(imagenet_stats, do_y=True))
    data.c = 3
    return data

def create_training_images(fn, i, p_hr, p_lr, size):
    """Create low quality images from folder p_hr in p_lr"""
    dest = p_lr/fn.relative_to(p_hr)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(fn)
    targ_sz = resize_to(img, size, use_min=True)
    img = img.resize(targ_sz, resample=Image.BILINEAR).convert('RGB')
    img.save(dest, quality=60) 


def do_fit(learn, epochs,save_name, lrs=slice(1e-3), pct_start=0.9):
    learn.fit_one_cycle(epochs, lrs, pct_start=pct_start)
    learn.save(save_name)
    learn.show_results(rows=1, imgsize=7)

path = Path('./dataset/')

path_fullRes = path/'DIV2K_train_HR'
path_lowRes = path/'DIV2K_train_LR_mild'

proj_id = 'unet_superRes_mobilenetV3_SSIM'

gen_name = proj_id + '_gen'
crit_name = proj_id + '_crit'

nf_factor = 2
pct_start = 1e-8

print("Dataset " + path_fullRes)

print("Creating unet with mobilenetv3")
model = geffnet.mobilenetv3_100

# # 256px

bs=10
sz=256
lr = 1e-2
wd = 1e-3
epochs = 5

data_gen = det_DIV2k_data(bs=bs, sz=sz)

print("Using SSIM Loss")
loss_func = msssim

learn_gen = gen_learner_wide(data=data_gen,
                             gen_loss=loss_func,
                             arch = model,
                             nf_factor=nf_factor)

wandbCallbacks = True

if wandbCallbacks:
    import wandb
    from wandb.fastai import WandbCallback
    config={"batch_size": bs,
            "img_size": (sz, sz),
            "learning_rate": lr,
            "weight_decay": wd,
            "num_epochs": epochs
    }
    wandb.init(project='SuperRes', config=config, id="gen_mobilenetV3_SSIM"+ datetime.now().strftime('_%m-%d_%H:%M'))

    learn_gen.callback_fns.append(partial(WandbCallback, input_type='images'))

print("Fitting with " + gen_name)
do_fit(learn_gen, epochs, gen_name+"_256px_0", slice(lr*10))

print("Unfreeze model")
learn_gen.unfreeze()

do_fit(learn_gen, 3, gen_name+"_256px_1", slice(lr))

bs=4
sz=512
epochs = 5

data_gen = det_DIV2k_data(bs, sz)

learn_gen.data = data_gen
learn_gen.freeze()
gc.collect()

learn_gen.load(gen_name+"_256px_1")

print("Upsize to gen_256_" + datetime.now().strftime('_%m-%d_%H:%M'))
do_fit(learn_gen, epochs, gen_name+"_512px_0")

learn_gen.unfreeze()

do_fit(learn_gen, epochs, gen_name+"_512px_1", slice(1e-6,1e-4), pct_start=0.3)

learn_gen = None
gc.collect()