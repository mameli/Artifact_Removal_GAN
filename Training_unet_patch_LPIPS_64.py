import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from pathlib import Path
from ArNet.metrics import *
from ArNet.ssim import *
from ArNet.fid_loss import *
from ArNet.save import *
from ArNet.loss import *
from ArNet.dataset import *
from ArNet.critics import *
from ArNet.generators import *
from fastai.vision import *
from fastai import *

from datetime import datetime
import geffnet  # efficient/ mobile net


def get_DIV2k_data_patches(pLow, bs: int):
    """Given the path of low resolution images with a proper suffix
       returns a databunch
    """
#     src = ImageImageList.from_folder(pLow, presort=True).split_by_idxs(
#         train_idx=list(range(0, 565407)), valid_idx=list(range(565408, 637408)))
    src = ImageImageList.from_folder(pLow, presort=True).split_by_rand_pct(valid_pct=0.05, seed=42)
    
    data = (src.label_from_func(lambda x: path_fullRes_patches/x.name)
            .transform(get_transforms(
                    flip_vert = True
                ), tfm_y=True
            ).databunch(bs=bs, num_workers=2, no_check=True).normalize(imagenet_stats, do_y=True))
    data.c = 3
    return data


def do_fit(learn, epochs, save_name, lrs=slice(1e-3), pct_start=0.3):
    learn.fit_one_cycle(epochs, lrs, pct_start=pct_start)
    learn.save("/data/students_home/fmameli/repos/Artifact_Removal_GAN/models/" + save_name)
    learn.show_results(rows=1, imgsize=10)


path = Path('./dataset/')

# path_fullRes_patches = path/"DIV2K_train_HR_Patches"/"64px_FullQF20"
# path_lowRes_patches = path/"DIV2K_train_LR_Patches"/"64px_1kQF20"

# # Flickr2K
# path_fullRes_patches = path/"Flickr2K"/'Flickr2K_HR_Patches'/"64px_FullQF20_Flickr"
# path_lowRes_patches = path/"Flickr2K"/'Flickr2K_LR_Patches'/"64px_512QF20_Flickr"

# Mixed
path_fullRes_patches = path/"DIV2K_Flickr_train_HR_Patches"/"64px_FullQF20_Flickr"
path_lowRes_patches = path/"DIV2K_Flickr_train_LR_Patches"/"64px_1kQF20_Flickr"


proj_id = 'unet_wideNf2_mobileMin_DivFlickr1k_P64px_SuperRes'


gen_name = proj_id + '_gen'
crit_name = proj_id + '_crit'

print(path_fullRes_patches)
print(path_lowRes_patches)
print(proj_id)
print("GPU usata ", torch.cuda.get_device_name())

# model = geffnet.mobilenetv3_rw
model = geffnet.mobilenetv3_small_minimal_100

loss_func = lpips_loss()

# # 64px patch

bs = 128
sz = 64
lr = 1e-3
wd = 1e-3
epochs = 1
nf_factor = 2

print("loading " + str(path_lowRes_patches) + " ...")
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
               id=proj_id + datetime.now().strftime('_%m-%d_%H_%M'))

    learn_gen.callback_fns.append(partial(WandbCallback, input_type='images'))

# print(learn_gen.summary())
# weights = "/data/students_home/fmameli/repos/Artifact_Removal_GAN/models/unet_wideNf2_superRes_mobilenetV3_GAN_best"
# learn_gen.load(weights, with_opt=False)
# learn_gen.load("/data/students_home/fmameli/repos/Artifact_Removal_GAN/wandb/run-20200515_110129-unet_wideNf2_superRes_mobilenetMinimal_1k_P64px_VGG_SuperRes_05-15_13_01/bestmodel", with_opt=False)

do_fit(learn_gen, 3, gen_name+"_0", slice(1e-2))
do_fit(learn_gen, 2, gen_name+"_1", slice(1e-3))
do_fit(learn_gen, 1, gen_name+"_2", slice(1e-4))

learn_gen.unfreeze()

do_fit(learn_gen, 1, gen_name+"_3", slice(1e-3), pct_start=0.001)
