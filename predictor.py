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
    return avg + A*x


def getRMSE(data,prediction):
    result = 0
    for i in range(len(data)):
        r = data[i]
        if (r < 1):
            r = 1
        elif (r > 5):
            r = 5
        if (i < len(prediction)):
            result += (r-prediction[i])**2
        else:
            result += r**2
    return np.around(np.sqrt(result/len(data)),3)
def getAbsError(data,prediction):
    return np.abs(data-prediction)