import time
import math
import pandas as pd
import numpy as np
import torch
import random
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from helper.distance_function import CONFIG_DISTANCE_FUNC

class kNN:
    def __init__(self, numNeighbors, trainPath, distFuncName, batchSize=1024, seed=18520185,numFolds=5):
        self.seed = seed
        self.numFolds = numFolds
        # Make seed
        random.seed(self.seed)
        self.numNeighbors = numNeighbors
        self.trainPath = trainPath
        self.batchSize = batchSize
        self.distanceFunc = self._getDistanceFunc(distFuncName)
        start = time.time()
        self.rawData = self._loadCSV()
        print(f'Load Training CSV time: {time.time() - start}')
        self.trainData, self.trainLabel = self.rawData[:,1:], self.rawData[:,0]

    def setNumNeighbors(self,numNeighbors):
        self.numNeighbors = numNeighbors

    def _loadCSV(self) -> torch.Tensor:
        trainDf = pd.read_csv(self.trainPath)
        pixelColumns = list(filter(lambda x: 'pixel' in x, trainDf.columns))
        trainData = trainDf[['label'] + pixelColumns]
        return torch.tensor(trainData.values,dtype=torch.float64)

    def _getDistanceFunc(self, distFuncName):
        distanceFunc = CONFIG_DISTANCE_FUNC[distFuncName]
        if not distanceFunc:
            raise Exception(f'Can not find any function name: {distFuncName}')
        return distanceFunc
    
    def _predictBatch(self, batchSamples):
        trainData, trainLabel = self.trainData, self.trainLabel
        distances = self.distanceFunc(trainData, batchSamples)
        _, batchLabelIds = torch.topk(distances,self.numNeighbors, largest=False)
        assert len(batchLabelIds) == len(batchSamples), 'Bug!!'
        predictions = list()
        # For each test sample in batch
        for _, neighborLabelIds in enumerate(batchLabelIds):
            neighborLabels = [trainLabel[i] for i in neighborLabelIds]
            # Voting with same weights
            weights = [1] * len(neighborLabels)
            occurrences = np.bincount(neighborLabels, weights)
            predictedLabel = np.argmax(occurrences)
            predictions.append(predictedLabel)
        return predictions

    def predict(self, testSamples):
        if not torch.is_tensor(testSamples):
            testSamples = torch.tensor(testSamples,dtype=torch.float64)
        predictions = list()
        # Compute each batch
        for batchId in tqdm(range(0,testSamples.size(0),self.batchSize)):
            batchSamples = testSamples[batchId:batchId+self.batchSize]
            batchOutput = self._predictBatch(batchSamples)
            predictions.extend(batchOutput)
        return predictions

    def evaluate(self):
        # Step 1: Split StratifiedKFold
        trainData, trainLabel = self.rawData[:,1:], self.rawData[:,0]
        skf = StratifiedKFold(n_splits=self.numFolds)
        scores = []
        for fold ,(trainIndex, valIndex) in enumerate(skf.split(trainData, trainLabel)):
            print(f'[INFO] Fold {fold} is running...')
            self.trainData, valData = trainData[trainIndex], trainData[valIndex]
            self.trainLabel, valLabel = trainLabel[trainIndex], trainLabel[valIndex]
            predictions = self.predict(valData)
            # Compute accuracy for each fold
            score = accuracy_score(predictions,valLabel)
            scores.append(score)
        # print(scores)
        return scores

    @staticmethod
    def loadTestSet(path):
        testDf = pd.read_csv(path)
        pixelColumns = list(filter(lambda x: 'pixel' in x, testDf.columns))
        testData = testDf[pixelColumns].to_numpy()
        return testData

if __name__ == '__main__':
    trainPath, testPath = './data/train.csv', './data/test.csv'
    myKNN = kNN(20,trainPath,'cosine')
    # testData = kNN.loadTestSet(testPath)
    # print(f'Train set size: {len(myKNN.trainData)}')
    # print(f'Test set size: {len(testData)}')
    # predictions = myKNN.predict(testData)
    # records = [(i+1,prediction) for i, prediction in enumerate(predictions)]
    # submissionDf = pd.DataFrame(records, columns=['ImageId','Label'])
    # submissionDf.to_csv('submission.csv',index=False)  
    scores = myKNN.evaluate()
    print(scores)
