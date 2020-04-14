import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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


# %%
def do_fit(learn, epochs, save_name, lrs=slice(1e-3), pct_start=0.3):
    learn.fit_one_cycle(epochs, lrs, pct_start=pct_start)
    learn.save("/data/students_home/fmameli/repos/SuperRes/models/" + save_name)
    learn.show_results(rows=1, imgsize=5)


# %%
def get_video_data(pLow, pFull, bs: int, sz: int):
    src = ImageImageList.from_folder(pLow, presort=True).split_by_rand_pct(0.1)
    data = (src.label_from_func(lambda x: path_ori/(x.name.replace("_VHS", ""))).transform(
        get_transforms(
        ),
        size=sz,
        tfm_y=True,
    ).databunch(bs=bs, num_workers=8, no_check=True)
        .normalize(imagenet_stats, do_y=True))
    data.c = 3
    return data


# %%
path = Path('./dataset/')

path_vhs = path / "Apple_vhs_videos/"
path_ori = path / "Apple_original_videos/"


# %%
data = get_video_data(path_vhs, path_ori, 2, 512)


# %%
data.show_batch(rows=1, ds_type=DatasetType.Train, imgsize=10)


# %%
proj_id = 'unet_vhs_lpips'

gen_name = proj_id + '_gen'

print(proj_id)

# %%
model = geffnet.mobilenetv3_rw

loss_func = lpips_loss()

bs = 3
sz = 512
lr = 1e-3
wd = 1e-3
epochs = 1
nf_factor = 2

data_gen = get_video_data(path_vhs, path_ori, 2, sz)
print("Dataset loaded...")

learn_gen = gen_learner_wide(data=data_gen,
                             gen_loss=loss_func,
                             arch=model,
                             nf_factor=nf_factor)


learn_gen.metrics.append(SSIM_Metric_gen())
learn_gen.metrics.append(SSIM_Metric_input())


# %%
data_gen


# %%
weights = "/data/students_home/fmameli/repos/SuperRes/models/unet_wideNf2_superRes_mobilenetV3_GAN_best"
learn_gen.load(weights, with_opt=False)


# %%
do_fit(learn_gen, 1, gen_name+"_0", slice(1e-2))


