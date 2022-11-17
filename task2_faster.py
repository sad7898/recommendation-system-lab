from matplotlib import pyplot as plt
import numpy as np
import scipy.sparse
import scipy.sparse.linalg as linalg
import scipy.spatial.distance as ssd
import predictor
import pandas as pd
import random
from load_data import load_data,getA

TPATH = "/courses/TSKS33/ht2022/data/student_files"
training_data = load_data("./verification.training")
test_data = load_data("./verification.test")

model,x,avg = predictor.getBaseLinePrediction(training_data)
prediction_test = predictor.testModel(test_data,x,avg)
def getCSR(data,prediction):
    # get row x col = movie x user csr matrix
    row = data[:,1]
    col = data[:,0]
    d = data[:,2]-prediction
    return np.array(scipy.sparse.csr_matrix((d,(row,col))).toarray())
    
def calculateCosineSimilarity(csr,minCommonUser):
    (row,_) = csr.shape
    resultRow = []
    resultCol = []
    resultData = []
    for i in range(row):
        for j in range(i,row):
            movie1 = csr[i]
            movie2 = csr[j]
            commonUserIndx = np.logical_and(movie1 > 0, movie2 > 0)
            if (len(movie1[commonUserIndx]) > minCommonUser):
                resultRow.append(i)
                resultCol.append(j)
                resultData.append(1-ssd.cosine(movie1[commonUserIndx],movie2[commonUserIndx]))
    return np.array(scipy.sparse.csr_matrix((resultData,(resultRow,resultCol)),shape=(row,row)).toarray())


def getImprovedPrediction(data,csr,csrSim,L):
    result = []
    for d in data:
        movie1 = d[1]
        user = d[0]
        simArr = csrSim[movie1]
        prediction = d[3]
        sim = sorted([(simArr[j],j) for j in range(len(simArr))],key=lambda k: k[0],reverse=True)[0:L]
        sumResidualCosSim = 0
        correction = 0
        sumAbs = sum([abs(b) for b in simArr]) 
        if (sumAbs > 0):
            for s in sim:
                movie2 = s[1]
                cosSim = s[0]
                residual = csr[movie2][user]
                sumResidualCosSim+=residual*cosSim
            correction = sumResidualCosSim/sumAbs
        result.append(prediction+correction)
    return result
dataArr = np.insert(training_data,len(training_data[0]),model,axis=1)
testDataArr = np.insert(test_data,len(test_data[0]),prediction_test,axis=1)
csr = getCSR(training_data,model)
csr_test = getCSR(test_data,prediction_test)
similarity = calculateCosineSimilarity(csr,50)
improvedPrediction = getImprovedPrediction(testDataArr,csr_test,similarity,100)
print("RMSE of test data: {0}".format(predictor.getRMSE(test_data[:,2],improvedPrediction)))








