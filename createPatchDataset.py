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
path_lowRes_Full = path/'DIV2K_train_LR_Full_QF20'


def get_patches(tensor):
    pw, ph = 32, 32
    nc, w, h = tensor.shape
    padW = pw - w % pw
    padH = ph - h % ph
    paded = F.pad(tensor, (padH, 0, padW, 0), value=1)
    patches = paded.unfold(0, nc, nc).unfold(1, ph, ph).unfold(2, pw, pw)
    return patches


high_res = ImageList.from_folder(path_fullRes, presort=True)
low_res = ImageList.from_folder(path_lowRes_Full, presort=True)

destHR = path/"DIV2K_train_HR_Patches"/"32px_FullQF20"
destHR.mkdir(parents=True, exist_ok=True)

destLR = path/"DIV2K_train_LR_Patches"/"32px_FullQF20"
destLR.mkdir(parents=True, exist_ok=True)

for index, img in enumerate(low_res):
    file_name = str(index+1).zfill(3)
    print("Creating file: " + file_name)
    resizedImg = img.resize(high_res[index].shape).clone()
    patches = get_patches(resizedImg.data)[0]
    row, col = patches.shape[:2]
    for i in range(row):
        for j in range(col):
            tempImg = Image(patches[i][j])
            fn = file_name + f'_patch_{i}_{j}.png'
            tempImg.save(destLR/fn)
