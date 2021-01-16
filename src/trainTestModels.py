import deepModels
import inputOutput
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.models import load_model
from keras.utils import np_utils

############################################################################################################
def createCallbacks(modelFile, earlyStoppingPatience):
    checkpoint = ModelCheckpoint(modelFile, monitor = 'val_loss', verbose = 1, save_best_only = True, 
                                 save_weights_only = True, mode = 'auto', period = 1)
    earlystopping = EarlyStopping(patience = earlyStoppingPatience, monitor = 'val_loss', verbose = 1, 
                                  restore_best_weights = True, min_delta=0.00001)
    #monitor = 'val_categorical_accuracy'
    return [checkpoint, earlystopping]
############################################################################################################
def plotConvergencePlots(hist, modelFile):
    fig = plt.figure()
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc = 'upper left')
    plotFile = modelFile[:-5] + '.png'
    plt.savefig(plotFile)
    plt.close(fig)
############################################################################################################
def trainModel(modelType, modelFile, trainInputList, trainOutputList, validInputList, validOutputList, 
               taskWeights, noOfFeatures, dropoutRate, layerNum = 4, 
               outputChannelNos = [1], outputTypes = ['C'], earlyStoppingPatience = 50, batchSize = 1, maxEpoch = 1000):
    inputHeight = trainInputList[0].shape[1]
    inputWidth = trainInputList[0].shape[2]
    channelNo = trainInputList[0].shape[3]

    print("trainInputList Shape: %s" % str(trainInputList[0].shape))
    print("trainOutputList Shape: %s" % str(trainOutputList[0].shape))
    
    if modelType == 'unet':
        model = deepModels.unet(inputHeight, inputWidth, channelNo, outputChannelNos, outputTypes, 
                                layerNum, noOfFeatures, dropoutRate, taskWeights)
    model.summary()
    
    hist = model.fit(x = trainInputList[0], y = trainOutputList[0][..., np.newaxis], 
                     validation_data = (validInputList[0], validOutputList[0][..., np.newaxis]), 
                     shuffle = True, batch_size = batchSize, epochs = maxEpoch, verbose = 1,
                     callbacks = createCallbacks(modelFile, earlyStoppingPatience))
    plotConvergencePlots(hist, modelFile)
############################################################################################################
def loadModel(modelType, modelFile, testInput, taskWeights, noOfFeatures, dropoutRate, 
              layerNum = 4, outputChannelNos = [1], outputTypes = ['C']):
    inputHeight = testInput.shape[1]
    inputWidth = testInput.shape[2]
    channelNo = testInput.shape[3]
    
    if modelType == 'unet':
        model = deepModels.unet(inputHeight, inputWidth, channelNo, outputChannelNos, outputTypes, 
                                layerNum, noOfFeatures, dropoutRate, taskWeights)
        
    model.load_weights(modelFile)
    return model
############################################################################################################
def testModel(model, testInput, weightMapNo = 1):
    # testInputList = []
    # testInputList.append(testInput)
    # for i in range(weightMapNo):
    #     weightNullMap = np.ones((testInput.shape[0], testInput.shape[1], testInput.shape[2]))
    #     testInputList.append(weightNullMap)
    print(testInput.shape)
    predictions = model.predict(testInput, batch_size = 4, verbose = 1)
    return predictions
############################################################################################################
