# -*- coding: utf-8 -*-
"""Project2_resnet34_bs8.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mqxs6xbApNAwZUapR1d3ifbNwiTFlETB
"""

from google.colab import files
from google.colab import drive

drive.mount('/content/drive')

"""Esse notebook é uma implementação do código acessado em: https://www.kaggle.com/dipam7/image-segmentation-using-fastai/notebook"""

# Commented out IPython magic to ensure Python compatibility.
# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

from fastai.vision import *
from fastai.callbacks.hooks import *

# Dataset: CAMVID - http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/

path = untar_data(URLs.CAMVID)  
path.ls()

path_lbl = path/'labels'
path_img = path/'images'

fnames = get_image_files(path_img)

lbl_names = get_image_files(path_lbl)

get_y_fn = lambda x: path_lbl/f'{x.stem}_P{x.suffix}'

img_f = fnames[0]
mask = open_mask(get_y_fn(img_f))

src_size = np.array(mask.shape[1:])
src_size,mask.data

size = src_size//2

codes = np.loadtxt(path/'codes.txt', dtype=str)

src = (SegmentationItemList.from_folder(path_img)
       .split_by_fname_file('../valid.txt')   # faz o slipt do dataset a partir dos nomes de imagens do arquivo de validação
       .label_from_func(get_y_fn, classes=codes))

data = (src.transform(get_transforms(), size=size, tfm_y=True)
        .databunch(bs=8, num_workers=0)
        .normalize(imagenet_stats))

name2id = {v:k for k,v in enumerate(codes)}
void_code = name2id['Void']

def acc_camvid(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()

metrics=acc_camvid

wd=1e-2 
# Weight decay

learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd)

lr_find(learn)
learn.recorder.plot()

learn.fit_one_cycle(10, slice(1e-06,1e-03), pct_start=0.8)

learn.save('resnet34_learnrate1')

learn.export('/content/drive/My Drive/resnet34_learnrate1.pkl')

learn.show_results(rows=2, figsize=(8,9))

learn.unfreeze()

lr_find(learn)
learn.recorder.plot()

learn.fit_one_cycle(10, slice(1e-5,1e-4), pct_start=0.8)

learn.save('resnet34_learnrate2')

learn.export('/content/drive/My Drive/resnet34_learnrate2.pkl')

learn.show_results(rows=2, figsize=(8,9))

learn.unfreeze()

lr_find(learn)
learn.recorder.plot()

learn.fit_one_cycle(10, slice(1e-4,1e-3), pct_start=0.8)

learn.save('resnet34_learnrate3')

learn.export('/content/drive/My Drive/resnet34_learnrate3.pkl')

learn.show_results(rows=2, figsize=(8,9))