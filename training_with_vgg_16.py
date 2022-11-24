import sys
import torch
import torchvision
from fastai.vision import *
from fastai.metrics import error_rate
from fastai.vision.data import ImageDataBunch
# from fastai.vision.data import ImageDataLoaders
from fastai import *
import cv2 as cv
import numpy as np
import pandas as pd
import scipy.io as sio
# from fastai.data import get_transform
import warnings
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
warnings.filterwarnings("ignore")


# Reference taken for the code
# dhamvi01(2019),FastAI-Image-Classification
# https://github.com/dhamvi01/FastAI-Image-Classification/blob/master/fastai.ipynb

# Data Processing


data = ImageDataBunch.from_folder('cars_classification_dataset', 'train', 'valid', ds_tfms=get_transforms(
    do_flip=False, flip_vert=True, max_rotate=5.0, max_zoom=1.1, max_lighting=0.2, max_warp=0.2, p_affine=0.75, p_lighting=0.75), size=224, bs=32).normalize(imagenet_stats)


# To show the batch images
data.show_batch()


# class names and number of classes
print(data.classes)
l = len(data.classes)
print(l)


# Training
# For the training, using VGG16 and batch size of 32.

epochs = 40

learn = cnn_learner(data, models.vgg16_bn, metrics=accuracy)
learn.fit_one_cycle(epochs)


data = learn.recorder.plot_losses()
data_acc = learn.recorder.plot_metrics()

steps_per_epoch = int(len(data[1])/epochs)
# print(steps_per_epoch)
training_loss = []
for st in range(epochs+1):
    if st == 0:
        training_loss.append(data[1][st])
    else:
        training_loss.append(data[1][(st*steps_per_epoch)-1])

total_epochs = [i for i in range(epochs+1)]
epochs_val = total_epochs.copy()
del epochs_val[0]
val_loss = data[-1]


figure, axis = plt.subplots(2, 2)
# For Train Loss
axis[0, 0].plot(total_epochs, training_loss, label='Train Loss')
axis[0, 0].set_title("Train Loss")
# For Test Loss
axis[0, 1].plot(epochs_val, val_loss, label='Test Loss')
axis[0, 1].set_title("Test Loss")
# For Test Loss
axis[1, 0].plot(epochs_val, data_acc, label='Test Accuracy')
axis[1, 0].set_title("Test Accuracy")
# For Overall Model Evaluation
axis[1, 1].plot(total_epochs, training_loss, label='Train Loss')
axis[1, 1].plot(epochs_val, val_loss, label='Test Loss')
axis[1, 1].plot(epochs_val, data_acc, label='Test Accuracy')
axis[1, 1].set_title("All Training Plots")
axis[1, 1].legend()
plt.savefig('vgg16_fastai_plot.png')

# calculate the accuracy
preds, y, loss = learn.get_preds(with_loss=True)
acc = accuracy(preds, y)
print('The accuracy is {0} %.'.format(acc))


# Exporting & saving model
interp = ClassificationInterpretation.from_learner(learn)

learn.save('stanford-cars-1')
learn.export('export.pkl')
