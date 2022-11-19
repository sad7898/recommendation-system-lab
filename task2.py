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
trainingData= load_data("{0}/sirko805.training".format(TPATH))
testData= load_data("{0}/sirko805.test".format(TPATH))

pTraining,model,avg = predictor.getBaseLinePrediction(trainingData)
pTest = predictor.testModel(testData,model,avg)
def getDataMatrix(data,prediction):
    # get row x col = movie x user csr matrix
    row = data[:,1] # row index (movie id)
    col = data[:,0] # col index (user id)
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


def getImprovedPrediction(data,csr,similarities,L):
    result = []
    for d in data:
        movie1 = d[1]
        user = d[0]
        simArr = similarities[movie1]
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

    
csr = getDataMatrix(trainingData,pTraining)
csr_test = getDataMatrix(testData,pTest)
similarity = calculateCosineSimilarity(csr,50)
ratingsTest = np.insert(testData,len(testData[0]),pTest,axis=1)
improvedPredictionTest = getImprovedPrediction(ratingsTest,csr_test,similarity,100)

print("RMSE of prediction test: {0}".format(predictor.getRMSE(testData[:,2],pTest)))
    
print("RMSE of improved prediction test: {0}".format(predictor.getRMSE(testData[:,2],improvedPredictionTest)))








