from pathlib import Path
from fastai.vision import *
from fastai import *
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# from PIL import Image, ImageDraw, ImageFont
# from PIL import ImageFile


path = Path('./dataset/')

path_fullRes = path/'DIV2K_train_HR'

path_lowRes_256 = path/'DIV2K_train_LR_256_QF20'
path_lowRes_512 = path/'DIV2K_train_LR_512_QF20'
path_lowRes_1k = path/'DIV2K_train_LR_1024_QF20'
path_lowRes_Full = path/'DIV2K_train_LR_Full_QF20'

# Flickr
path_fullRes_flickr = path/'Flickr2K'/ 'Flickr2K_HR'

path_lowRes_512_mixed = path/'Flickr2K'/'Flickr2K_LR_512_QF20'
path_lowRes_1024_mixed = path/'Flickr2K'/'Flickr2K_LR_512_QF20'

# Mixed
path_fullRes_mixed = path/'DIV2K_Flickr_Mixed_HR'

path_lowRes_512_mixed = path/'DIV2K_Flickr_Mixed_LR_512_QF20'
path_lowRes_1024_mixed = path/'DIV2K_Flickr_Mixed_LR_1024_QF20'

def get_patches(tensor):
    pw, ph = 64, 64
    nc, w, h = tensor.shape
    padW = pw - w % pw
    padH = ph - h % ph
    paded = F.pad(tensor, (padH, 0, padW, 0), value=1)
    patches = paded.unfold(0, nc, nc).unfold(1, ph, ph).unfold(2, pw, pw)
    return patches


# high_res = ImageList.from_folder(path_fullRes, presort=True)
# low_res = ImageList.from_folder(path_lowRes_Full, presort=True)
# low_res  = ImageList.from_folder(path_lowRes_512, presort=True) # super resolution
# low_res  = ImageList.from_folder(path_lowRes_1k, presort=True) # super resolution

# Flickr
# high_res = ImageList.from_folder(path_fullRes_flickr, presort=True)
# low_res  = ImageList.from_folder(path_lowRes_512_mixed, presort=True)

# Mixed
high_res = ImageList.from_folder(path_fullRes_mixed, presort=True)
low_res  = ImageList.from_folder(path_lowRes_1024_mixed, presort=True)

destHR = path/"DIV2K_Flickr_train_HR_Patches"/"64px_FullQF20_Flickr"
destHR.mkdir(parents=True, exist_ok=True)

destLR = path/"DIV2K_Flickr_train_LR_Patches"/"64px_1kQF20_Flickr"
destLR.mkdir(parents=True, exist_ok=True)

print("Creating " + str(destHR) + "folder..." )
for index, img in enumerate(high_res):
    file_name = str(index+1).zfill(6)
    print("Creating file: High res " + file_name)
    patches = get_patches(img.data)[0]
    row, col = patches.shape[:2]
    for i in range(row):
        for j in range(col):
            tempImg = Image(patches[i][j])
            fn = file_name + f'_patch_{i}_{j}.png'
            tempImg.save(destHR/fn)

print("Creating " + str(destLR) + "folder..." )
for index, img in enumerate(low_res):
    file_name = str(index+1).zfill(6)
    print("Creating file: Low res " + file_name)
    resizedImg = img.resize(high_res[index].shape).clone()
    patches = get_patches(resizedImg.data)[0]
    row, col = patches.shape[:2]
    for i in range(row):
        for j in range(col):
            tempImg = Image(patches[i][j])
            fn = file_name + f'_patch_{i}_{j}.png'
            tempImg.save(destLR/fn)
