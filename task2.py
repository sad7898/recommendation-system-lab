from matplotlib import pyplot as plt
import numpy as np
import scipy.sparse.linalg as linalg
import scipy.spatial.distance as ssd
import predictor
import pandas as pd
from load_data import load_data,getA

TPATH = "/courses/TSKS33/ht2022/data/student_files"
training_data = load_data("./verification.training")
test_data = load_data("./verification.test")

def getResidual(data,prediction):
    return data[:,2]-prediction
def getCommonUserResidualsIndx(users1,users2):
    result = []
    for indx1,user1 in enumerate(users1):
        indx2 = -1
        try:
            indx2 = users2.index(user1)
            result.append((indx1,indx2))
        except ValueError:
            continue
    return result

model,x,avg = predictor.getBaseLinePrediction(training_data)
prediction_test = predictor.testModel(training_data,x,avg)

residual = getResidual(training_data,prediction_test)
dataArr = np.insert(training_data,len(training_data[0]),residual,axis=1)
df = pd.DataFrame(dataArr,columns=['user','movie','rating','residual'])
groupedDf = df.groupby('movie').apply(lambda df,user,residual: [df[user],df[residual]],"user","residual").reset_index(name="data")
groupedDf["cosSim"] = np.empty((len(groupedDf),0)).tolist()
# get residual 
minimumCommonUser = 10
L = 10
for row1 in groupedDf.itertuples():
    i = row1.Index
    movie1 = row1.movie
    data1 = row1.data
    users1 = list(data1[0])
    residuals1 = list(data1[1])
    if (len(users1) < minimumCommonUser):
        continue
    if (i < groupedDf.size-1):   
        for row2 in groupedDf[i+1:].itertuples():
            j = row2.Index
            movie2 = row2.movie
            data2 = row2.data
            users2 = list(data2[0])
            residuals2 = list(data2[1])
            if (len(users2) < minimumCommonUser):
                continue
            commonUserResidual = getCommonUserResidualsIndx(users1,users2)
            if (len(commonUserResidual) > minimumCommonUser):
                ri = [residuals1[cu[0]] for cu in commonUserResidual] 
                rj = [residuals2[cu[1]] for cu in commonUserResidual]
                cosSim = 1-ssd.cosine(ri,rj)
                cosSimAtI = groupedDf.at[i,"cosSim"]
                cosSimAtJ = groupedDf.at[j,"cosSim"]
                groupedDf.at[i,"cosSim"].append((cosSim,j))
                groupedDf.at[j,"cosSim"].append((cosSim,i))
print("done with big loop")
groupedDf= groupedDf.assign(cosSim=groupedDf.apply(lambda row: sorted(row[2],key=lambda e: e[0])[:L],axis=1))
groupedDf = groupedDf.drop("data",axis=1)
improvedModel = []
def getImprovedPrediction(row,similarities,data):
    # col = 'user','movie','rating','residual','prediction'
    absSum = 0
    sumProdResidual = 0
    user = row[0]
    prediction = row[4]
    for sim in similarities:
        (cosSim,simMovie) = sim
        absSum += abs(cosSim)
        residual = data[np.logical_and(simMovie==data[:,1],user==data[:,0])]
        if (len(residual) > 0):
            sumProdResidual += residual[0]*cosSim
    return prediction+sumProdResidual/absSum
p = np.insert(dataArr,len(dataArr[0]),model,axis=1)
processedDf = pd.DataFrame(p,columns=['user','movie','rating','residual','prediction'])
print("calculating improved prediction")
for row in p:
    improvedModel.append(getImprovedPrediction(row,groupedDf.at[row[1],"cosSim"],p))


print(improvedModel[:10])
prediction_test = predictor.testModel(training_data,improvedModel,avg)
print("RMSE of test data: {0}".format(predictor.getRMSE(test_data[:,2],prediction_test)))





                







# choose a pair of movie i,j
# count number of users that rated i,j if count > 10 then
# get residual of ratings of common users that rate i an j
# calculate cos sim
# sort top 10 cos sim for each movie
# 

# [movie,[[user,residual],[user2,residual2]]]




