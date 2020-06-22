from fastai.vision.transform import get_transforms
from fastai.vision.data import ImageDataBunch, ImageImageList, imagenet_stats
from fastai.vision.data import resize_to, ImageList
from fastai.core import partial, parallel
from pathlib import Path
import PIL


def get_databunch(
    sz: int,
    bs: int,
    crappy_path: Path,
    good_path: Path,
    random_seed: int = 42,
    keep_pct: float = 1.0,
    num_workers: int = 8,
    xtra_tfms=[],
) -> ImageDataBunch:

    src = (
        ImageImageList.from_folder(crappy_path, convert_mode='RGB')
        .use_partial_data(sample_pct=keep_pct, seed=random_seed)
        .split_by_rand_pct(0.1, seed=random_seed)
    )

    data = (
        src.label_from_func(lambda x: good_path / x.relative_to(crappy_path))
        .transform(
            get_transforms(
                max_zoom=2.
            ),
            size=sz,
            tfm_y=True,
        )
        .databunch(bs=bs, num_workers=num_workers, no_check=True)
        .normalize(imagenet_stats, do_y=True)
    )

    data.c = 3
    return data


def get_data(pLow, pFull, bs: int, sz: int, keep_pct: float):
    return get_databunch(pLow, pFull, sz=sz, bs=bs,
                         random_seed=None, keep_pct=keep_pct)


def get_DIV2k_data(pLow, pFull, bs: int, sz: int):
    """Given the path of low resolution images with a proper suffix
       returns a databunch
    """
    suffixes = {"dataset/DIV2K_train_LR_x8": "x8",
                "dataset/DIV2K_train_LR_difficult": "x4d",
                "dataset/DIV2K_train_LR_mild": "x4m"}
    lowResSuffix = suffixes[str(pLow)]
    src = ImageImageList.from_folder(pLow, presort=True).split_by_idxs(
        train_idx=list(range(0, 800)), valid_idx=list(range(800, 900)))

    data = (src.label_from_func(
        lambda x: pFull/(x.name).replace(lowResSuffix, '')
    ).transform(
        get_transforms(
            max_rotate=30,
            max_zoom=3.,
            max_lighting=.4,
            max_warp=.4,
            p_affine=.85
        ),
        size=sz,
        tfm_y=True,
    ).databunch(bs=bs, num_workers=8, no_check=True)
        .normalize(imagenet_stats, do_y=True))
    data.c = 3
    return data


def get_DIV2k_data_QF(pLow, pFull, bs: int, sz: int):
    """Given the path of low resolution images
       returns a databunch
    """
    src = ImageImageList.from_folder(pLow, presort=True).split_by_idxs(
        train_idx=list(range(0, 800)), valid_idx=list(range(800, 900)))

    data = (src.label_from_func(
        lambda x: pFull/(x.name.replace(".jpg", ".png"))
    ).transform(
        get_transforms(
            max_zoom=2.
        ),
        size=sz,
        tfm_y=True
    ).databunch(bs=bs, num_workers=8, no_check=True)
        .normalize(imagenet_stats, do_y=True))
    data.c = 3
    return data

def get_dummy_databunch(bs: int, sz: int):
    """Returns sz databunch
    """
    path = Path('./dataset/dummy/')
    src = ImageImageList.from_folder(path).split_none()

    data = (src.label_from_func(
        lambda x: path/(x.name.replace(".jpg", ".png"))
    ).transform(
        size=sz,
        tfm_y=True
    ).databunch(bs=bs, num_workers=1, no_check=True)
        .normalize(imagenet_stats, do_y=True))
    data.c = 3
    return data


def create_training_images(fn, i, p_hr, p_lr, size, qualityFactor, downsize=True):
    """Create low quality images from folder p_hr in p_lr"""
    dest = p_lr/fn.relative_to(p_hr)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = PIL.Image.open(fn)
    if downsize:
        targ_sz = resize_to(img, size, use_min=True)  # W x H
        img = img.resize(targ_sz, resample=PIL.Image.BILINEAR).convert('RGB')
    img.save(dest.with_suffix(".jpg"), "JPEG", quality=qualityFactor)


def create_dataset(path_fullRes: Path, path_list, downsize=True):
    il = ImageList.from_folder(path_fullRes)

    for p, size, qf in path_list:
        if not p.exists():
            print(f"Creating {p}")
            print(f"Size: {size} with {qf} quality factor")
            parallel(partial(create_training_images,
                             p_hr=path_fullRes,
                             p_lr=p,
                             size=size,
                             qualityFactor=qf,
                             downsize=downsize), il.items)
