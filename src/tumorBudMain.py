import inputOutput
import deepModels
import os
import numpy as np
import sys, getopt

import tumorBudTasks
############################################################################################################
def callUnet(modelType, pathPrefix, layerNum, featNum, dropoutRate, runNo, trainStr, outTypes, residual):
    networkName = modelType + '_tumor_bud'
    outNos = [2]
    taskWeights = [1.0]

    if trainStr == 'tr':
        tumorBudTasks.trainUnetForTumorBudDetection(modelType, pathPrefix, networkName, runNo, layerNum, 
                                                    featNum, dropoutRate, outNos, outTypes, taskWeights,residual)
    else:
        [probs, actual, tsNames, resPath] = tumorBudTasks.testUnetForTumorBudDetection(modelType, pathPrefix, 
                                                    networkName, runNo, layerNum, featNum, dropoutRate, 
                                                    outNos, outTypes, taskWeights)
        labels = inputOutput.findLabels(probs)
        inputOutput.saveSegmentationLabels(resPath, tsNames, labels, '_lb')
        inputOutput.saveProbabilities(resPath, tsNames, probs[:,:,:,1], '_pr')
        return [probs, actual]
############################################################################################################
def main(argv):
    # nohup python tumorBudMain.py 6 1 tr ./TB_detection/ 5 C >> out1 &
    # nohup python tumorBudMain.py 6 1 ts ./TB_detection/ 5 C >> out1 &

    pathPrefix = argv[3]
    layerNum = int(argv[4])
    featNum = 32
    dropoutRate = 0.2
    
    gpu = argv[0]
    runNo = int(argv[1])
    trainStr = argv[2]
    outTypes = [argv[5]]
    residual = True

    #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    #os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    
    callUnet('unet', pathPrefix, layerNum, featNum, dropoutRate, runNo, trainStr, outTypes, residual)
############################################################################################################
if __name__ == "__main__":
   main(sys.argv[1:])
############################################################################################################
