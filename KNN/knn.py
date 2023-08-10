import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
def solution(path):  
   data=loadmat(path) # loading data from NumberRecognition.mat file
   traindata_eight = data['imageArrayTraining8']  # selecting training and testing arrays of Eight and Nine
   traindata_nine = data['imageArrayTraining9']
   testdata_eight=data['imageArrayTesting8']
   testdata_nine=data['imageArrayTesting9']
   # Appending eight and nine arrays
   X_train = np.append(traindata_eight, traindata_nine,2)  # (28,28,1500)
   X_train = X_train.transpose(2,0,1)   #(1500,28,28)
   X_trainingset = X_train.reshape(X_train.shape[0],np.prod(X_train.shape[1:]))
   X_trainingset.shape
   #creating training labels
   X_label=np.append(np.zeros(traindata_eight.shape[-1]),np.ones(traindata_nine.shape[-1]))
   # Appending testing arrays of Eight and Nine
   X_test=np.append(testdata_eight,testdata_nine,2) #(28,28,500)
   X_test=X_test.transpose(2,0,1) #(500,28,28)
   # Reshaping the testing set
   Y_testingset=X_test.reshape(X_test.shape[0],np.prod(X_test.shape[1:]))
   #creating testing labels
   Y_label=np.append(np.zeros(testdata_eight.shape[-1]),np.ones(testdata_nine.shape[-1]))
   #Finding error rates using kNN Classifier
   accuracy=[]
   errors=[]
   for k in range(1,21): # for k=20
      Knn_classifier = KNeighborsClassifier(n_neighbors = k)
      Knn_classifier.fit(X_trainingset, X_label)
      pred = Knn_classifier.predict(Y_testingset)
      accuracy.append(sklearn.metrics.accuracy_score(pred,Y_label)) # Calculating the accuracies 
      errors.append((np.mean(pred!=Y_label))*100) # Calculating error rate
   #plotting graph for the error rates
   plt.figure(figsize=(8, 6))
   plt.plot(range(1, 21), errors, color='grey', linestyle='dashed', marker='o', markerfacecolor='grey', markersize=8)
   plt.title('Error Rate K Value')
   plt.xlabel('K Value')
   plt.ylabel('Error')
   plt.show()


if __name__ == "__main__":
   solution(r"C:\Users\Nikhila reddy\NumberRecognition.mat")