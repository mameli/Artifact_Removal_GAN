import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from pathlib import Path
from superRes.metrics import *
from superRes.ssim import *
from superRes.fid_loss import *
from superRes.save import *
from superRes.loss import *
from superRes.dataset import *
from superRes.critics import *
from superRes.generators import *
from fastai.vision import *
from fastai import *

from datetime import datetime
import geffnet  # efficient/ mobile net


def get_DIV2k_data_patches(pLow, bs: int):
    """Given the path of low resolution images with a proper suffix
       returns a databunch
    """
    src = ImageImageList.from_folder(pLow, presort=True).split_by_idxs(
        train_idx=list(range(0, 565407)), valid_idx=list(range(565408, 637408)))
    data = (src.label_from_func(lambda x: path_fullRes_patches/x.name)
            .databunch(bs=bs, num_workers=8, no_check=True).normalize(imagenet_stats, do_y=True))
    data.c = 3
    return data


def do_fit(learn, epochs, save_name, lrs=slice(1e-3), pct_start=0.9):
    learn.fit_one_cycle(epochs, lrs, pct_start=pct_start)
    learn.save(save_name)
    learn.show_results(rows=1, imgsize=10)


path = Path('./dataset/')

path_fullRes_patches = path/"DIV2K_train_HR_Patches"/"64px_FullQF20"
path_lowRes_patches = path/"DIV2K_train_LR_Patches"/"64px_FullQF20"

path_fullRes = path/'DIV2K_train_HR'
path_lowRes_256 = path/'DIV2K_train_LR_256_QF20'
path_lowRes_512 = path/'DIV2K_train_LR_512_QF20'
path_lowRes_Full = path/'DIV2K_train_LR_Full_QF20'

proj_id = 'unet_wideNf3_superRes_mobilenetV3_Patches64px'

gen_name = proj_id + '_gen'
crit_name = proj_id + '_crit'

nf_factor = 3

print(path_fullRes_patches)
print(proj_id)
print("GPU usata ", torch.cuda.get_device_name())

model = geffnet.mobilenetv3_rw

loss_func = lpips_loss()

# # 64px patch

bs = 100
sz = 64
lr = 1e-3
wd = 1e-3
epochs = 1


data_gen = get_DIV2k_data_patches(path_lowRes_patches, bs=bs)
print("Dataset loaded...")

learn_gen = gen_learner_wide(data=data_gen,
                             gen_loss=loss_func,
                             arch=model,
                             nf_factor=nf_factor)


learn_gen.metrics.append(SSIM_Metric_gen())
learn_gen.metrics.append(SSIM_Metric_input())

wandbCallbacks = True 

if wandbCallbacks:
    import wandb
    from wandb.fastai import WandbCallback
    config = {"batch_size": bs,
              "img_size": (sz, sz),
              "learning_rate": lr,
              "weight_decay": wd,
              "num_epochs": epochs
              }
    wandb.init(project='SuperRes', config=config,
               id="unet_wideNf3_superRes_mobilenetV3_Patches64px" + datetime.now().strftime('_%m-%d_%H:%M'))

    learn_gen.callback_fns.append(partial(WandbCallback, input_type='images'))

do_fit(learn_gen, 1, gen_name+"_64px_0", 1e-3)


learn_gen.unfreeze()


do_fit(learn_gen, 5, gen_name+"_64px_2", 1e-3)
