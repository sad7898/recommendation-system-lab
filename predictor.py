import numpy as np
import scipy.sparse.linalg as linalg
from load_data import load_data,getA
TPATH = "/courses/TSKS33/ht2022/data/student_files"
training_data = load_data("{0}/sirko805.training".format(TPATH))

def getAvgRating(data):
    return np.mean(data[:,2])
def getBaseLinePrediction(data):
    avg = getAvgRating(data)
    y = data[:,2]-avg
    A = getA(data)
    x = linalg.lsqr(A,y)[0]
    return [
        avg + A*x, 
        x,
        avg
    ]

def testModel(data,x,avg):
    A = getA(data)
    return avg+A*x

def getRMSE(data,prediction):
    result = 0
    minLength = min(len(prediction),len(data))
    for i in range(minLength):
        r = data[i]
        p = prediction[i]
        if (p < 1):
            p = 1
        elif (p > 5):
            p = 5 
        result += (r-p)**2
    return np.around(np.sqrt(result/minLength),3)
def getAbsError(data,prediction):
    return np.abs(data-prediction)