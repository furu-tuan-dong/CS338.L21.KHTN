import numpy as np
import argparse
from kNN import kNN

def run(opt):
    print(opt)
    k = np.arange(1,20)
    model = kNN(0,opt.training_csv,opt.dist_func,opt.batch_size,numFolds=opt.num_folds)

    kScores = list()
    for i in k:
        print(f'[INFO] Setting k = {i}')
        model.setNumNeighbors(i)
        scores = model.evaluate()
        meanScore = np.mean(scores)
        print(f'[INFO] Mean score: {meanScore}')
        kScores.append(meanScore)
    maxIndex = np.argmax(kScores)
    print(f'Best K: {k[maxIndex]}, Best score: {kScores[maxIndex]}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_csv', type=str, default="./data/train.csv", help='Training CSV file')
    parser.add_argument('--dist_func', type=str, default='euclid', help='Choose a metric')
    parser.add_argument('--batch_size', type=int, default=2048, help='set batch size')
    parser.add_argument('--num_folds', type=int, default=3, help='Set number of folds')
    opt = parser.parse_args()
    run(opt)