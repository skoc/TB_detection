import inputOutput
import deepModels
import os
import numpy as np
import sys, getopt, glob
############################################################################################################
def evaluateOneSegmentationLabels(segPath, annotPath):
    labels = inputOutput.loadSegmentationLabelTxt(segPath)
    labels = inputOutput.loadProbabilitiesTxt(segPath) > 0.5
    annot = inputOutput.loadSegmentationLabelImage(annotPath)

    total = labels.shape[0] * labels.shape[1]
    accuracy = np.sum(np.bitwise_xor(labels, annot))/total
    tp = np.sum(np.bitwise_and(labels, annot))

    if np.sum(labels) == 0:
        precision = 0
    else:
        precision = tp / np.sum(labels)
    
    if np.sum(annot) == 0:
        recall = 0
    else:
        recall = tp / np.sum(annot)

    if precision + recall == 0:
        f1score = 0
    else:
        f1score = (2*precision*recall)/(precision+recall)
    return accuracy, precision, recall, f1score
############################################################################################################
def evaluateOneSet(segPath, annotPath):
    images = glob.glob(os.path.join(segPath, '*_pr'))
    print("Total number of read images: ", len(images))
    accuracies = []
    precisions = []
    recalls = []
    f1scores = []
    
    counter = 0
    print("Filename, Accuracy, Precision, Recall, F1Score")
    for segImagePath in images:
        basename = os.path.basename(segImagePath)[:-3]
        annotImagePath = os.path.join(annotPath, basename+'.png')
        if os.path.exists(segImagePath) and os.path.exists(annotImagePath):
            [accuracy, precision, recall, f1score] = evaluateOneSegmentationLabels(segImagePath, annotImagePath)
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1scores.append(f1score)
            print(os.path.basename(segImagePath) + " {:2.2%}, {:2.2%}, {:2.2%}, {:2.2%}".format(accuracy, precision, recall, f1score))
            counter +=1 

    print("Average values: {:2.2%}, {:2.2%}, {:2.2%}, {:2.2%}".format(np.mean(accuracies), np.mean(precisions), np.mean(recalls), np.mean(f1scores)))
    print("Total number of processed images: ", counter)
############################################################################################################
def main(argv):
    segPrefix = argv[0]
    annotPrefix = argv[1]
    evaluateOneSet(segPrefix, annotPrefix)
############################################################################################################
if __name__ == "__main__":
   main(sys.argv[1:])
############################################################################################################
