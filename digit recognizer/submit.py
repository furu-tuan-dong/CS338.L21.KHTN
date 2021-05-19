import pandas as pd
import math, requests
from kNN import kNN

def submit_prediction(df, sep=',', comment='', compression='gzip', **kwargs):
    TOKEN='97f1f48cfc06e3ebeff1e7f986211c44cf8ffbefec537f655a534dab9e260475fb34752e04ca23c9530752885d9bc5bb2b87a15e2ae498860278f2d093051098'
    URL='http://submission.mmlab.uit.edu.vn/api/submissions'
    df.to_csv('temporary.dat', sep=sep, compression=compression, **kwargs)
    r = requests.post(URL, headers={'Authorization': 'Bearer {}'.format(TOKEN)},files={'datafile': open('temporary.dat', 'rb')},data={'comment':comment, 'compression': compression})
    if r.status_code == 429:
        raise Exception('Submissions are too close. Next submission is only allowed in {} seconds.'.format(int(math.ceil(int(r.headers['x-rate-limit-remaining']) / 1000.0))))
    if r.status_code != 200:
        raise Exception(r.text)
        

if __name__ == '__main__':
    trainPath, testPath = './data/train.csv', './data/test.csv'
    myKNN = kNN(3,trainPath,'euclid')
    testData = kNN.loadTestSet(testPath)
    print(f'Train set size: {len(myKNN.trainData)}')
    print(f'Test set size: {len(testData)}')
    predictions = myKNN.predict(testData)
    records = [(i+1,prediction) for i, prediction in enumerate(predictions)]
    submissionDf = pd.DataFrame(records, columns=['ImageId','Label'])
    submissionDf.to_csv('submission.csv',index=False)  
    submit_prediction(submissionDf, sep=',', index=True, comment='my submission')
