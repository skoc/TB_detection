import tensorflow as tf
from keras import backend as K
import numpy as np

from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, Dropout, concatenate
from keras.layers.core import Layer, Dense, Activation, Flatten, Reshape, Permute, Lambda
from keras import objectives, optimizers, losses
from keras.utils import np_utils



############################################################################################################
# Region-based Loss Functions
############################################################################################################
# Dice Loss
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_weights):
    def loss_fn(y_true, y_pred):
        loss = 1-dice_coef(y_true, y_pred)
        return K.mean(loss)
    return loss_fn
############################################################################################################
# Tversky Loss
"""
By setting the value of α >b, you can penalise false negatives more. 
In the case where α =b = 0.5, it simplifies into the dice coefficient.
Let's try is with α = 0.3, b = 0.7
"""
def tversky(y_true, y_pred, smooth=1):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.3
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_weights):
    def loss_fn(y_true, y_pred):
        return K.mean(1 - tversky(y_true,y_pred))
    return loss_fn
############################################################################################################
# Focal Tversky Loss
"""
In the case of class imbalance, the FTL becomes useful when γ > 1.
This results in a higher loss gradient for examples where TI < 0.5. 
This forces the model to focus on harder examples,
especially small scale segmentations which usually receive low TI scores.
if γ = 1, it will be standard Tversky Loss
Let's try is with α = 0.3, b =0.7, γ = 2
"""
def focal_tversky(y_weights):
    def loss_fn(y_true, y_pred):
        pt_1 = tversky(y_true, y_pred)
        gamma = 2.0
        return K.pow((1-pt_1), gamma)
    return loss_fn
############################################################################################################
# Distributation-based Loss Functions
############################################################################################################
# Focal Loss
def focal_loss(y_weights):
    def loss_fn(y_true, y_pred):
        gamma, alpha = 2.5, 0.25 # these values work best accord. paper
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.mean((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

    return loss_fn

############################################################################################################
def weighted_binary_crossentropy(y_true, y_pred, y_weight):
    bce = losses.binary_crossentropy(y_true, y_pred)
    loss = bce * y_weight
    return K.mean(loss)
############################################################################################################
def weighted_categorical_crossentropy(y_weight):
    def loss(y_true, y_pred):        
        cce = objectives.categorical_crossentropy(y_true, y_pred)
        loss = (y_weight * cce)
        return K.mean(loss)
    return loss
############################################################################################################
def weighted_mse(y_weight):
    def loss(y_true, y_pred):
        mse = objectives.mean_squared_error(y_true, y_pred)
        loss = (y_weight * mse)
        return K.mean(loss)
    return loss
############################################################################################################
def weighted_sparse_categorical(y_weight):
    def loss(y_true, y_pred):
        mse = objectives.sparse_categorical_crossentropy(y_true[:,:,:,1], y_pred)
        loss = (y_weight * mse)
        return K.mean(loss)
    return loss
############################################################################################################
def weighted_hinge(y_weight):
    def loss(y_true, y_pred):
        h = objectives.hinge(y_true, y_pred)
        loss = (y_weight * h)
        return K.mean(loss)
    return loss
############################################################################################################
def weighted_poisson(y_weight):
    def loss(y_true, y_pred):
        h = objectives.poisson(y_true, y_pred)
        loss = (y_weight * h)
        return K.mean(loss)
    return loss
############################################################################################################
def kl_divergence(y_weight):
    def loss(y_true, y_pred):
        loss = objectives.kullback_leibler_divergence(y_true, y_pred)
        return K.mean(loss)
    return loss
############################################################################################################
def calculateClassWeights(gold):
    totalPixels = gold.shape[0] * gold.shape[1]
    classNo = int(gold.max() + 1)
    classCounts = np.zeros(classNo)
    classWeights = np.zeros(classNo)
    for i in range(classNo):
        classCounts[i] = (gold == i).sum()
        classWeights[i] = classCounts[i] / totalPixels

    # 1 / freq, sum of the class weights = 1
    total = 0
    for i in range(classNo):
        if classWeights[i] > 0:
            classWeights[i] = 1 / classWeights[i]
            total += classWeights[i]
            
    for i in range(classNo):
        classWeights[i] /= total

    #classWeights[-1] *= 10

    return classWeights
############################################################################################################        
def calculateClassWeightMapOneImage(gold):
    classWeights = calculateClassWeights(gold)

    inputHeight = gold.shape[0]
    inputWidth = gold.shape[1]
    
    lossWeightMap = np.zeros((inputHeight, inputWidth))
    for i in range(inputHeight):
        for j in range(inputWidth):
            lossWeightMap[i][j] = classWeights[int(gold[i][j])]
            
    # normalized such that sum of the loss weights = image resolution
    lossWeightMap = (inputHeight * inputWidth * lossWeightMap) / lossWeightMap.sum()
            
    return lossWeightMap
############################################################################################################
def classWeightMaps(golds):
    imageNo = golds.shape[0]
    inputHeight = golds.shape[1]
    inputWidth = golds.shape[2]

    imageWeights = np.zeros((imageNo, inputHeight, inputWidth))
    for i in range(imageNo):
        imageWeights[i] = calculateClassWeightMapOneImage(golds[i])
        
    return imageWeights
############################################################################################################
def trValWeights(trOutputs, valOutputs, lossType):
    if lossType == 'same':
        trWeights = np.ones((trOutputs.shape[0], trOutputs.shape[1], trOutputs.shape[2]))
        valWeights = np.ones((valOutputs.shape[0], valOutputs.shape[1], valOutputs.shape[2]))
    elif lossType == 'class-weighted':
        trWeights = classWeightMaps(trOutputs)
        valWeights = classWeightMaps(valOutputs)
    return [trWeights, valWeights]
############################################################################################################


