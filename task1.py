from matplotlib import pyplot as plt
import numpy as np
import predictor
from load_data import load_data,getA
TPATH = "/courses/TSKS33/ht2022/data/student_files"
training_data = load_data("{0}/sirko805.training".format(TPATH))
test_data = load_data("{0}/sirko805.test".format(TPATH))
prediction = predictor.getBaseLinePrediction(training_data)
print("RMSE of training data: {0}".format(predictor.getRMSE(training_data[:,2],prediction)))
print("RMSE of test data: {0}".format(predictor.getRMSE(test_data[:,2],prediction)))
abs_errors = predictor.getAbsError(test_data[:,2],np.around(prediction[0:len(test_data)]))
plt.hist(abs_errors, bins=[0,1,2,3,4,5],label="absolute error in rating prediction")
plt.show()

