import numpy as np
from keras.utils import np_utils
import os
from os import listdir
import errno
import cv2

############################################################################################################
def listAllJPG(imageNames, datasetPath, prefix):
    fileList = listdir(datasetPath + 'data/mask/')
    for i in range(len(fileList)):
        if fileList[i][-4::] == '.png':
            if len(prefix) == '0':
                imageNames.append(fileList[i][:-4])
            elif fileList[i][0:len(prefix)] == prefix:
                imageNames.append(fileList[i][:-4])
    return imageNames
############################################################################################################
def normalizeImage(img):
    normImg = np.zeros(img.shape)
    for i in range(img.shape[2]):
        if(img[:, :, i].std() != 0):
            normImg[:, :, i] = (img[:, :, i] - img[:, :, i].mean()) / (img[:, :, i].std())
            
    return normImg
############################################################################################################
def createCategoricalOutput(gold, binaryClass):
    if binaryClass:
        gold = gold > 0

    categoricalOutput = np_utils.to_categorical(gold)
    return categoricalOutput
############################################################################################################
def findLabels(probs):
    imageNo = probs.shape[0]
    inputHeight = probs.shape[1]
    inputWidth = probs.shape[2]
    
    labels = np.zeros((imageNo, inputHeight, inputWidth))
    for i in range(len(probs)):
        labels[i] = np.argmax(probs[i], axis = 2)
    return labels
############################################################################################################
def readOneTumorBudImage(datasetPath, imageName, goldFlag):
    imageFileName = datasetPath + 'data/tile/' + imageName + '.png'
    segmFileName = datasetPath + 'data/mask/' + imageName + '.png'
    
    img = cv2.imread(imageFileName)
    img = img[...,::-1] # convert from BGR to RGB
    img = normalizeImage(img)
    
    if (goldFlag):
        gold = cv2.imread(segmFileName)
        gold = gold[:, :, 0]
        gold = gold > 0
        return [img, gold]
    else:
        return img
############################################################################################################
def readOneTumorBudDataset(datasetPath, imageNames, goldFlag):
    d_names = []
    d_inputs = []
    d_outputs = []
    
    for fname in imageNames:
        if (goldFlag):
            [img, gold] = readOneTumorBudImage(datasetPath, fname, True)
            d_outputs.append(gold)
        else:
            img = readOneTumorBudImage(datasetPath, fname, False)
        d_inputs.append(img)
        d_names.append(fname)

    d_inputs = np.asarray(d_inputs)
    if (goldFlag):
        d_outputs = np.asarray(d_outputs)
        return [d_inputs, d_outputs, d_names]
    else:
        return [d_inputs, d_names]
############################################################################################################
def makeDirectory(directoryPath):
    try:
        os.mkdir(directoryPath) 
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
############################################################################################################
def saveProbabilities(dirPath, names, probs, postfix):
    print('Saving to : ', dirPath)
    makeDirectory(dirPath)
    
    for i in range(len(probs)):
        fname = dirPath + '/' + names[i] + postfix
        np.savetxt(fname, probs[i], fmt = '%1.4f')
    print('Probabilities are saved')
############################################################################################################
def saveSegmentationLabels(dirPath, names, labels, postfix):
    print('Saving to : ', dirPath)
    makeDirectory(dirPath)
    
    for i in range(len(labels)):
        fname = dirPath + '/' + names[i] + postfix
        np.savetxt(fname, labels[i], fmt = '%1.0f')
    print('Labels are saved')
############################################################################################################
def loadSegmentationLabelImage(imagePath):
    im = cv2.imread(imagePath)
    im = im[:, :, 0]>0
    return im
############################################################################################################
def loadSegmentationLabelTxt(txtPath):
    im = np.loadtxt(txtPath) > 0
    return im
############################################################################################################
def loadProbabilitiesTxt(txtPath):
    im = np.loadtxt(txtPath)
    return im
