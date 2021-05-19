import torch

def EuclideanDistance(trainSamples, testSamples):
        return torch.cdist(testSamples, trainSamples, p=2)

def CosineDistance(trainSamples, testSamples):
    trainSamplesNorm = trainSamples / trainSamples.norm(dim=1)[:, None]
    testSamplesNorm = testSamples / testSamples.norm(dim=1)[:, None]
    return 1 - testSamplesNorm.matmul(trainSamplesNorm.T)


CONFIG_DISTANCE_FUNC = {
    'euclid': EuclideanDistance,
    'cosine': CosineDistance
}