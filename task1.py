from matplotlib import pyplot as plt
import numpy as np
import predictor
from load_data import load_data,getA
TPATH = "/courses/TSKS33/ht2022/data/student_files"
training_data = load_data("./verification.training")
test_data = load_data("./verification.test")
prediction,x,avg = predictor.getBaseLinePrediction(training_data)
prediction_test = predictor.testModel(test_data,x,avg)
print("RMSE of training data: {0}".format(predictor.getRMSE(training_data[:,2],prediction)))
print("RMSE of test data: {0}".format(predictor.getRMSE(test_data[:,2],prediction_test)))
abs_errors = predictor.getAbsError(test_data[:,2],np.around(np.clip(prediction_test,1,5)))
plt.hist(abs_errors, bins=[0,1,2,3,4,5],label="absolute error in rating prediction")
plt.show()

