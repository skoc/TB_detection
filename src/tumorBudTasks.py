import inputOutput
import trainTestModels
import calculateLoss
import os
import numpy as np
from keras.utils import np_utils

############################################################################################################
def returnTumorBudTaskInfo(pathPrefix, networkName, layerNum, featNum, runNo):
    modelName = networkName + '_L' + str(layerNum) + '_F' + str(featNum) + '_run' + str(runNo)
    modelPath = pathPrefix + 'models/' + modelName + '.hdf5'
    resultPath = pathPrefix + 'results/' + modelName
    
    trNames = inputOutput.listAllJPG([], pathPrefix, 'OTC-1-')
    trNames = inputOutput.listAllJPG(trNames, pathPrefix, 'OTC-8-')
    trNames = inputOutput.listAllJPG(trNames, pathPrefix, 'OTC-11-')
    
    valNames = inputOutput.listAllJPG([], pathPrefix, 'OTC-10-')
    
    tsNames = inputOutput.listAllJPG([], pathPrefix, 'OTC-12-')
    tsNames = inputOutput.listAllJPG(tsNames, pathPrefix, 'OTC-88-')
    # To check the overfit
    tsNames = inputOutput.listAllJPG(tsNames, pathPrefix, 'OTC-10-')
    
    return [modelPath, resultPath, trNames, valNames, tsNames]
############################################################################################################
def taskLists4UNet(pathPrefix, trNames, valNames):
    [trInputs, trOutputs, trNames] = inputOutput.readOneTumorBudDataset(pathPrefix, trNames, True)    
    [valInputs, valOutputs, valNames] = inputOutput.readOneTumorBudDataset(pathPrefix, valNames, True)

    # [trWeights, valWeights] = calculateLoss.trValWeights(trOutputs, valOutputs, 'class-weighted')
    #[trWeights, valWeights] = calculateLoss.trValWeights(trOutputs, valOutputs, 'same')
    
    # trOutputs = inputOutput.createCategoricalOutput(trOutputs, True)
    # valOutputs = inputOutput.createCategoricalOutput(valOutputs, True)
    
    trInputList = [trInputs]#, trWeights]
    valInputList = [valInputs]#, valWeights]
    trOutputList = [trOutputs]
    valOutputList = [valOutputs]
    
    return [trInputList, trOutputList, valInputList, valOutputList]
############################################################################################################
def trainUnetForTumorBudDetection(modelType, pathPrefix, networkName, runNo, layerNum, 
                                  featNum, dropoutRate, outNos, outTypes, taskWeights):
    [modelPath, _, trNames, valNames, _] = returnTumorBudTaskInfo(pathPrefix, networkName, layerNum, featNum, runNo)
    [trInList, trOutList, valInList, valOutList] = taskLists4UNet(pathPrefix, trNames, valNames)
    trainTestModels.trainModel(modelType, modelPath, trInList, trOutList, valInList, valOutList, 
                               taskWeights, featNum, dropoutRate, layerNum, outNos, outTypes)
############################################################################################################
def testUnetForTumorBudDetection(modelType, pathPrefix, networkName, runNo, layerNum, 
                                 featNum, dropoutRate, outNos, outTypes, taskWeights):
    [modelPath, resultPath, _, _, tsNames] = returnTumorBudTaskInfo(pathPrefix, networkName, layerNum, featNum, runNo)    
    [tsInputs, tsOutputs, tsNames] = inputOutput.readOneTumorBudDataset(pathPrefix, tsNames, tsNames)
    model = trainTestModels.loadModel(modelType, modelPath, tsInputs, taskWeights, featNum, dropoutRate, 
                                      layerNum, outNos, outTypes) 
    probs = trainTestModels.testModel(model, tsInputs, len(outNos))
    return [probs, tsOutputs, tsNames, resultPath]
############################################################################################################
