import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from fastai import *
from fastai.vision import *
from superRes.generators import *
from superRes.critics import *
from superRes.dataset import *
from superRes.loss import *
from superRes.save import *
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageFile
from datetime import datetime

import geffnet # efficient/ mobile net
def get_data(bs:int, sz:int, keep_pct:float):
    return get_databunch(sz=sz, bs=bs, crappy_path=path_lowRes, 
                         good_path=path_fullRes, 
                         random_seed=None, keep_pct=keep_pct)

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
    learn.show_results(rows=1, imgsize=5)

path = untar_data(URLs.PETS)

path_fullRes = path/'images'
path_lowRes = path/'lowRes-96'
path_medRes = path/'lowRes-256'

proj_id = 'unet_superRes_resnet34'

gen_name = proj_id + '_gen'
crit_name = proj_id + '_crit'

TENSORBOARD_PATH = Path('data/tensorboard/' + proj_id)

nf_factor = 2
pct_start = 1e-8

print(path_fullRes)

sets = [(path_lowRes, 96),(path_medRes, 256)]
il = ImageList.from_folder(path_fullRes)

for p,size in sets:
    if not p.exists():
        print(f"resizing to {size} into {p}")
        parallel(partial(create_training_images, p_hr=path_fullRes, p_lr=p, size=size), il.items)

print("Creating unet with resnet34")
model = models.resnet34

# # 128px

bs=10
sz=128
lr = 1e-3
wd = 1e-3
keep_pct=1.0
epochs = 10

data_gen = get_data(bs=bs, sz=sz, keep_pct=keep_pct)

learn_gen = gen_learner_wide(data=data_gen,
                             gen_loss=FeatureLoss(),
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
    wandb.init(project='SuperRes', config=config, id="gen_resnet34"+ datetime.now().strftime('_%m-%d_%H:%M'))

    learn_gen.callback_fns.append(partial(WandbCallback, input_type='images'))

# learn_gen.lr_find()
# learn_gen.recorder.plot()
# learn_gen.summary()

print("Fitting with " + gen_name)
do_fit(learn_gen, epochs, gen_name+"_128px_0", slice(lr*10))

print("Unfreeze model")
learn_gen.unfreeze()

do_fit(learn_gen, epochs, gen_name+"_128px_1", slice(1e-5, lr))

sz=256

data_gen = get_data(bs, sz, keep_pct=keep_pct)

learn_gen.data = data_gen
learn_gen.freeze()
gc.collect()

learn_gen.load(gen_name+"_128px_1")

print("Upsize to gen_256_" + datetime.now().strftime('_%m-%d_%H:%M'))
do_fit(learn_gen, epochs, gen_name+"_256px_0")

learn_gen.unfreeze()

do_fit(learn_gen, epochs, gen_name+"_256px_1", slice(1e-6,1e-4), pct_start=0.3)

learn_gen = None
gc.collect()