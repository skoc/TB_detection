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
    
    # New Data Split Update Added: 03/03/2021
    trNames = inputOutput.listAllJPG([], 'newdata/', 'OTC-13-')
    trNames = inputOutput.listAllJPG(trNames, 'newdata/', 'OTC-14-')
    trNames = inputOutput.listAllJPG(trNames, 'newdata/', 'OTC-23-')
    trNames = inputOutput.listAllJPG(trNames, 'newdata/', 'OTC-26-')
    trNames = inputOutput.listAllJPG(trNames, 'newdata/', 'OTC-27-')
    trNames = inputOutput.listAllJPG(trNames, 'newdata/', 'OTC-32-')
    trNames = inputOutput.listAllJPG(trNames, 'newdata/', 'OTC-38-')
    trNames = inputOutput.listAllJPG(trNames, 'newdata/', 'OTC-42-')
    trNames = inputOutput.listAllJPG(trNames, 'newdata/', 'OTC-98-')
    trNames = inputOutput.listAllJPG(trNames, 'newdata/', 'OTC-81-')
    trNames = inputOutput.listAllJPG(trNames, 'newdata/', 'OTC-64-')
    trNames = inputOutput.listAllJPG(trNames, 'newdata/', 'OTC-85-')
    trNames = inputOutput.listAllJPG(trNames, 'newdata/', 'OTC-8-')
    trNames = inputOutput.listAllJPG(trNames, 'newdata/', 'OTC-11-')
    trNames = inputOutput.listAllJPG(trNames, 'newdata/', 'OTC-10-')
    trNames = inputOutput.listAllJPG(trNames, 'newdata/', 'OTC-12-')
    trNames = inputOutput.listAllJPG(trNames, 'newdata/', 'OTC-88-')

    valNames = inputOutput.listAllJPG([], 'newdata/', 'OTC-53-')
    valNames = inputOutput.listAllJPG(valNames, 'newdata/', 'OTC-83-')
    valNames = inputOutput.listAllJPG(valNames, 'newdata/', 'OTC-76-')
    valNames = inputOutput.listAllJPG(valNames, 'newdata/', 'OTC-92-')
    valNames = inputOutput.listAllJPG(valNames, 'newdata/', 'OTC-93-')
    valNames = inputOutput.listAllJPG(valNames, 'newdata/', 'OTC-46-')
    valNames = inputOutput.listAllJPG(valNames, 'newdata/', 'OTC-152-')


    tsNames = inputOutput.listAllJPG([], 'newdata/', 'OTC-104-')
    tsNames = inputOutput.listAllJPG(tsNames, 'newdata/', 'OTC-107-')
    tsNames = inputOutput.listAllJPG(tsNames, 'newdata/', 'OTC-123-')
    tsNames = inputOutput.listAllJPG(tsNames, 'newdata/', 'OTC-129-')
    tsNames = inputOutput.listAllJPG(tsNames, 'newdata/', 'OTC-131-')
    tsNames = inputOutput.listAllJPG(tsNames, 'newdata/', 'OTC-134-')
    tsNames = inputOutput.listAllJPG(tsNames, 'newdata/', 'OTC-1-')


    return [modelPath, resultPath, trNames, valNames, tsNames]
############################################################################################################
def taskLists4UNet(pathPrefix, trNames, valNames):
    [trInputs, trOutputs, trNames] = inputOutput.readOneTumorBudDataset(pathPrefix, trNames, True)    
    [valInputs, valOutputs, valNames] = inputOutput.readOneTumorBudDataset(pathPrefix, valNames, True)

    #[trWeights, valWeights] = calculateLoss.trValWeights(trOutputs, valOutputs, 'class-weighted')
    [trWeights, valWeights] = calculateLoss.trValWeights(trOutputs, valOutputs, 'same')
    
    trOutputs = inputOutput.createCategoricalOutput(trOutputs, True)
    valOutputs = inputOutput.createCategoricalOutput(valOutputs, True)
    
    trInputList = [trInputs, trWeights]
    valInputList = [valInputs, valWeights]
    trOutputList = [trOutputs]
    valOutputList = [valOutputs]
    
    return [trInputList, trOutputList, valInputList, valOutputList]
############################################################################################################
def trainUnetForTumorBudDetection(modelType, pathPrefix, networkName, runNo, layerNum, featNum, dropoutRate, outNos, outTypes, taskWeights):
    [modelPath, _, trNames, valNames, _] = returnTumorBudTaskInfo(pathPrefix, networkName, layerNum, featNum, runNo)
    [trInList, trOutList, valInList, valOutList] = taskLists4UNet(pathPrefix, trNames, valNames)
    trainTestModels.trainModel(modelType, modelPath, trInList, trOutList, valInList, valOutList, taskWeights, featNum, dropoutRate, layerNum, outNos, outTypes)
############################################################################################################
def testUnetForTumorBudDetection(modelType, pathPrefix, networkName, runNo, layerNum, featNum, dropoutRate, outNos, outTypes, taskWeights):
    [modelPath, resultPath, _, _, tsNames] = returnTumorBudTaskInfo(pathPrefix, networkName, layerNum, featNum, runNo)    
    [tsInputs, tsOutputs, tsNames] = inputOutput.readOneTumorBudDataset(pathPrefix, tsNames, tsNames)
    model = trainTestModels.loadModel(modelType, modelPath, tsInputs, taskWeights, featNum, dropoutRate, layerNum, outNos, outTypes)
    probs = trainTestModels.testModel(model, tsInputs, len(outNos))
    return [probs, tsOutputs, tsNames, resultPath]
############################################################################################################
