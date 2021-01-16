import inputOutput
import deepModels
import os
import numpy as np
import sys, getopt

import tumorBudTasks
############################################################################################################
def callUnet(modelType, pathPrefix, layerNum, featNum, dropoutRate, runNo, trainStr, outTypes):
    networkName = modelType + '_tumor_bud'
    outNos = [1]
    taskWeights = [1.0]

    if trainStr == 'tr':
        tumorBudTasks.trainUnetForTumorBudDetection(modelType, pathPrefix, networkName, runNo, layerNum, 
                                                    featNum, dropoutRate, outNos, outTypes, taskWeights)
    else:
        [probs, actual, tsNames, resPath] = tumorBudTasks.testUnetForTumorBudDetection(modelType, pathPrefix, 
                                                    networkName, runNo, layerNum, featNum, dropoutRate, 
                                                    outNos, outTypes, taskWeights)
        labels = inputOutput.findLabels(probs)
        print(probs.shape)
        inputOutput.saveSegmentationLabels(resPath, tsNames, labels, '_lb')
        inputOutput.saveProbabilities(resPath, tsNames, probs, '_pr')
        return [probs, actual]
############################################################################################################
def main(argv):
    # nohup python tumorBudMain.py 6 1 tr >> out1 &
    # nohup python tumorBudMain.py 6 1 ts >> out1 &
    pathPrefix = argv[3]
    layerNum = 5
    featNum = 32
    dropoutRate = 0.2
    
    gpu = argv[0]
    runNo = int(argv[1])
    trainStr = argv[2]
    outTypes = [argv[4]]
    
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    
    callUnet('unet', pathPrefix, layerNum, featNum, dropoutRate, runNo, trainStr, outTypes)
############################################################################################################
if __name__ == "__main__":
   main(sys.argv[1:])
############################################################################################################
