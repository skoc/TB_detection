import tensorflow as tf
from keras import backend as K
import numpy as np

from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, Dropout, concatenate
from keras.layers.core import Layer, Dense, Activation, Flatten, Reshape, Permute, Lambda
from keras import objectives, optimizers, losses

import calculateLoss

############################################################################################################
def unetOneBlock(blockInput, noOfFeatures, filterSize, dropoutRate):
    blockOutput = Convolution2D(noOfFeatures, filterSize, activation = 'relu', padding = 'same', 
                                kernel_initializer = 'glorot_uniform')(blockInput)
    blockOutput = Dropout(dropoutRate)(blockOutput)
    blockOutput = Convolution2D(noOfFeatures, filterSize, activation = 'relu', padding = 'same', 
                                kernel_initializer = 'glorot_uniform')(blockOutput)
    return blockOutput
############################################################################################################
def unetOneEncoderBlock(blockInput, noOfFeatures, filterSize, dropoutRate):
    conv = unetOneBlock(blockInput, noOfFeatures, filterSize, dropoutRate)
    pool = MaxPooling2D(pool_size = (2, 2))(conv)
    return [conv, pool]
############################################################################################################
def oneEncoderPath4(inputs, noOfFeatures, filterSize, dropoutRate):
    [conv1, poolR] = unetOneEncoderBlock(inputs, noOfFeatures, filterSize, dropoutRate)
    [conv2, poolR] = unetOneEncoderBlock(poolR, 2 * noOfFeatures, filterSize, dropoutRate)
    [conv3, poolR] = unetOneEncoderBlock(poolR, 4 * noOfFeatures, filterSize, dropoutRate)
    [conv4, poolR] = unetOneEncoderBlock(poolR, 8 * noOfFeatures, filterSize, dropoutRate)
    return [conv1, conv2, conv3, conv4, poolR]
############################################################################################################
def oneEncoderPath5(inputs, noOfFeatures, filterSize, dropoutRate):
    [conv1, poolR] = unetOneEncoderBlock(inputs, noOfFeatures, filterSize, dropoutRate)
    [conv2, poolR] = unetOneEncoderBlock(poolR, 2 * noOfFeatures, filterSize, dropoutRate)
    [conv3, poolR] = unetOneEncoderBlock(poolR, 4 * noOfFeatures, filterSize, dropoutRate)
    [conv4, poolR] = unetOneEncoderBlock(poolR, 8 * noOfFeatures, filterSize, dropoutRate)
    [conv5, poolR] = unetOneEncoderBlock(poolR, 16 * noOfFeatures, filterSize, dropoutRate)
    return [conv1, conv2, conv3, conv4, conv5, poolR]
############################################################################################################
def unetOneDecoderBlock(blockInput, longSkipInput, noOfFeatures, filterSize, dropoutRate):
    upR = concatenate([UpSampling2D(size = (2, 2))(blockInput), longSkipInput], axis = 3)
    conv = unetOneBlock(upR, noOfFeatures, filterSize, dropoutRate)    
    return conv
############################################################################################################
def oneDecoderPath4(lastConv, conv4, conv3, conv2, conv1, noOfFeatures, filterSize, dropoutRate):
    deconv4 = unetOneDecoderBlock(lastConv, conv4, 8 * noOfFeatures, filterSize, dropoutRate)
    deconv3 = unetOneDecoderBlock(deconv4, conv3, 4 * noOfFeatures, filterSize, dropoutRate)
    deconv2 = unetOneDecoderBlock(deconv3, conv2, 2 * noOfFeatures, filterSize, dropoutRate)
    deconv1 = unetOneDecoderBlock(deconv2, conv1, noOfFeatures, filterSize, dropoutRate)
    return deconv1
############################################################################################################
def oneDecoderPath5(lastConv, conv5, conv4, conv3, conv2, conv1, noOfFeatures, filterSize, dropoutRate):
    deconv5 = unetOneDecoderBlock(lastConv, conv5, 16 * noOfFeatures, filterSize, dropoutRate)
    deconv4 = unetOneDecoderBlock(deconv5, conv4, 8 * noOfFeatures, filterSize, dropoutRate)
    deconv3 = unetOneDecoderBlock(deconv4, conv3, 4 * noOfFeatures, filterSize, dropoutRate)
    deconv2 = unetOneDecoderBlock(deconv3, conv2, 2 * noOfFeatures, filterSize, dropoutRate)
    deconv1 = unetOneDecoderBlock(deconv2, conv1, noOfFeatures, filterSize, dropoutRate)
    return deconv1
############################################################################################################
def outputLayer(lastDeconv, outputWeights, outputType, outputChannelNo, name):
    if outputType == 'C':
        outputs = Convolution2D(outputChannelNo, (1, 1), activation = 'sigmoid', name = name)(lastDeconv)
        outputLoss = calculateLoss.weighted_categorical_crossentropy(outputWeights)
    elif outputType == 'L':
        outputs = Convolution2D(outputChannelNo, (1, 1), activation = 'linear', name = name)(lastDeconv)
        outputLoss = calculateLoss.weighted_mse(outputWeights)
    elif outputType == 'S':
        outputs = Convolution2D(outputChannelNo, (1, 1), activation = 'sigmoid', name = name)(lastDeconv)
        outputLoss = calculateLoss.weighted_sparse_categorical(outputWeights)
    elif outputType == 'H':
        outputs = Convolution2D(outputChannelNo, (1, 1), activation = 'sigmoid', name = name)(lastDeconv)
        outputLoss = calculateLoss.weighted_hinge(outputWeights)
    elif outputType == 'P':
        outputs = Convolution2D(outputChannelNo, (1, 1), activation = 'sigmoid', name = name)(lastDeconv)
        outputLoss = calculateLoss.weighted_poisson(outputWeights)
    elif outputType == 'KL':
        outputs = Convolution2D(outputChannelNo, (1, 1), activation = 'sigmoid', name = name)(lastDeconv)
        outputLoss = calculateLoss.kl_divergence(outputWeights)
    elif outputType == 'FTV':
        outputs = Convolution2D(outputChannelNo, (1, 1), activation = 'sigmoid', name = name)(lastDeconv)
        outputLoss = calculateLoss.focal_tversky(outputWeights)
    elif outputType == 'TV':
        outputs = Convolution2D(outputChannelNo, (1, 1), activation = 'sigmoid', name = name)(lastDeconv)
        outputLoss = calculateLoss.tversky_loss(outputWeights)
    elif outputType == 'DC':
        outputs = Convolution2D(outputChannelNo, (1, 1), activation = 'sigmoid', name = name)(lastDeconv)
        outputLoss = calculateLoss.dice_coef_loss(outputWeights)
    elif outputType == 'F':
        outputs = Convolution2D(outputChannelNo, (1, 1), activation = 'sigmoid', name = name)(lastDeconv)
        outputLoss = calculateLoss.focal_loss(outputWeights)


    return [outputs, outputLoss]
############################################################################################################
def createInputOutputLostLists(inputHeight, inputWidth, inputs, 
                               lastDeconv1, lastDeconv2, lastDeconv3, lastDeconv4, 
                               outputChannelNos, outputTypes):
    inputList = []
    outputList = []
    lossList = []
    
    inputList.append(inputs)
    
    outputWeights1 = Input(shape = (inputHeight, inputWidth))
    [output1, loss1] = outputLayer(lastDeconv1, outputWeights1, outputTypes[0], outputChannelNos[0], 'out1')
    inputList.append(outputWeights1)
    outputList.append(output1)
    lossList.append(loss1)
    
    taskNo = len(outputChannelNos)
    if taskNo >= 2:
        outputWeights2 = Input(shape = (inputHeight, inputWidth))
        [output2, loss2] = outputLayer(lastDeconv2, outputWeights2, outputTypes[1], outputChannelNos[1], 'out2')
        inputList.append(outputWeights2)
        outputList.append(output2)
        lossList.append(loss2)
    if taskNo >= 3:
        outputWeights3 = Input(shape = (inputHeight, inputWidth))
        [output3, loss3] = outputLayer(lastDeconv3, outputWeights3, outputTypes[2], outputChannelNos[2], 'out3')
        inputList.append(outputWeights3)
        outputList.append(output3)
        lossList.append(loss3)
    if taskNo >= 4:
        outputWeights4 = Input(shape = (inputHeight, inputWidth))
        [output4, loss4] = outputLayer(lastDeconv4, outputWeights4, outputTypes[3], outputChannelNos[3], 'out4')
        inputList.append(outputWeights4)
        outputList.append(output4)
        lossList.append(loss4)

    return [inputList, outputList, lossList]
############################################################################################################
# supports multitask architecture upto four different tasks
def unet(inputHeight, inputWidth, channelNo, outputChannelNos, outputTypes, layerNum, noOfFeatures, dropoutRate, taskWeights):
    filterSize = (3, 3)
    optimizer = optimizers.Adadelta()

    inputs = Input(shape = (inputHeight, inputWidth, channelNo), name = 'input')
    if layerNum == 4:
        [conv1, conv2, conv3, conv4, poolR] = oneEncoderPath4(inputs, noOfFeatures, filterSize, dropoutRate)
        lastConv = unetOneBlock(poolR, 16 * noOfFeatures, filterSize, dropoutRate)
        lastDeconv1 = oneDecoderPath4(lastConv, conv4, conv3, conv2, conv1, noOfFeatures, filterSize, dropoutRate)
        lastDeconv2 = oneDecoderPath4(lastConv, conv4, conv3, conv2, conv1, noOfFeatures, filterSize, dropoutRate)
        lastDeconv3 = oneDecoderPath4(lastConv, conv4, conv3, conv2, conv1, noOfFeatures, filterSize, dropoutRate)
        lastDeconv4 = oneDecoderPath4(lastConv, conv4, conv3, conv2, conv1, noOfFeatures, filterSize, dropoutRate)
        
    elif layerNum == 5:
        [conv1, conv2, conv3, conv4, conv5, poolR] = oneEncoderPath5(inputs, noOfFeatures, filterSize, dropoutRate)
        lastConv = unetOneBlock(poolR, 32 * noOfFeatures, filterSize, dropoutRate)
        lastDeconv1 = oneDecoderPath5(lastConv, conv5, conv4, conv3, conv2, conv1, noOfFeatures, filterSize, dropoutRate)
        lastDeconv2 = oneDecoderPath5(lastConv, conv5, conv4, conv3, conv2, conv1, noOfFeatures, filterSize, dropoutRate)
        lastDeconv3 = oneDecoderPath5(lastConv, conv5, conv4, conv3, conv2, conv1, noOfFeatures, filterSize, dropoutRate)
        lastDeconv4 = oneDecoderPath5(lastConv, conv5, conv4, conv3, conv2, conv1, noOfFeatures, filterSize, dropoutRate)
        
    [inputList, outputList, lossList] = createInputOutputLostLists(inputHeight, inputWidth, inputs, lastDeconv1, lastDeconv2, 
                                                                   lastDeconv3, lastDeconv4, outputChannelNos, outputTypes)
    model = Model(inputs = inputList, outputs = outputList)
    model.compile(loss = lossList, loss_weights = taskWeights, optimizer = optimizer, metrics = [calculateLoss.dice_coef])
    return model
############################################################################################################
