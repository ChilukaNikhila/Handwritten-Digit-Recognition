<!--
###############################################################################
Overall Grade:       61 / 100
Responses:           15 / 40
Code:                46 / 60
###############################################################################
-->

#Question 1

##a 
As the K values increases number of the neighbours increases so the testing error rises. When the k value is low the number of neigbours is less compared to higher k value.The data points increases as the k values increases, so the distant ones have the impact on error rate.
<!--
    >>> JUEQI: 5 / 10
    ---------------------
    You have to mention specific aspects of the number recognition data, e.g. 8s and 9s are very similar in this the number recognition data set. With K increased, it use more samples from the wrong classes to predict.

-->

##b 
The error rate at the lowest value of k is zero. The error rate at lowest value is zero because the closest point to any training data point is itself. So, it will  always overfit.The error rate increaseswith the value of k. So we should try out different K values and plot error rate.
<!--
    >>> JUEQI: 3 / 10
    --------------------
    The answer should be: It is not reliable. The error rate at the lowest K is zero. This is not a reliable performance because an error rate of zero is very unlikely, and best explanation is there was no real class overlap (no really ambiguous 8s/9s). With more data, this would be less likely, as there should be more varied samples.

-->


#Question 2

##Data set Description 

The data set consists of placement data of students.It includes secondary and higher secondary school percentage,degree specialization, salary,work experience and type,status of the students if they are placed or not.

Following are the different columns in the data set
sl_no         : Gives the serial number
gender        : Describes the gender
ssc_p         : Secondary school is percentage
ssc_b         : sencondary school is central or others
hsc_p         : higher secondary school percentage
hsc_b         : high secondary school is central or others
hsc_s         : Higher secondary school course
degree_p      : Degree percentage
degree_t      : Degree type 
workex        : Gives workexperience
specialisation: Gives specialization
etest_p       : test percentage
mba_p         : mba percentage
status        : Gives status of student placed or Not placed
salary        : Gives the salary of the student
My group of interest is the status of the student Placed or Not placed which is considered as label

Here Dummies are used to convert categorical data of all the columns.

##AUC Values
Listing out the highest AUC values for the features

| Feature                | AUC        |
|:-----------------------|:----------:|
| ssc_p                  | 0.87384026 |
| degree_p               | 0.80849133 |
| hsc_p                  | 0.78882614 |
| workex_Yes             | 0.64158935 |
| specialisation_Mkt&Fin | 0.63437878 |
| etest_p                | 0.57104679 |
| gender_M               | 0.54679306 |
| mba_p                  | 0.53948165 |
| degree_t_Comm&Mgmt     | 0.52369907 |
| ssc_b_Others           | 0.52006858 |

<!--
    >>> JUEQI: 5 / 12
    ---------------------
    Didn't mention total number of samples, total number of measurements, sample counts for your group of interest and sample count for the group not of interest.

    You need to sort AUC values from strongest (most interesting, far from 0.5) to weakest (least interesting, near 0.5).


-->


#Question 3

#Dataset Description
The data set consists of placement data of students.It includes secondary and higher secondary school percentage,degree specialization, salary,work experience and type,status of the students if they are placed or not.My group of interest is the status of the student Placed or Not placed which is considered as label


Here for the given data set the error rate is high at the initial stages when compared to the  digit recognition example.
In this Placement data the error rate is high at the initial stages and decreased until some value of K.Then after decreasing, the error rate followed reverse V shape.The error increased and decreased as the k value increases.

But in the digit recognition example the error rate increased as the  value of K increased.In placement data of students the error rate didnot follow a particular trend.
<!--
    >>> JUEQI: 2 / 8
    ---------------------
    No real elaboration.
-->













<!--
====================================================================
Automatic Output:
isort:
--- /home/jq/Desktop/ML_TA/assign1/CSCI 597/Nikhila Chiluka_620274_assignsubmission_file_/assign1.py:before	2021-10-14 16:02:07.412711
+++ /home/jq/Desktop/ML_TA/assign1/CSCI 597/Nikhila Chiluka_620274_assignsubmission_file_/assign1.py:after	2021-10-14 16:02:18.885925
@@ -7,14 +7,16 @@
 ###############################################################################
 """
 
+import matplotlib.pyplot as plt
 import numpy as np
 import pandas as pd
+import sklearn
 from scipy.io import loadmat
-import matplotlib.pyplot as plt
-import sklearn
+from sklearn.model_selection import train_test_split
 from sklearn.neighbors import KNeighborsClassifier
-from sklearn.model_selection import train_test_split
 from sklearn.preprocessing import LabelEncoder
+
+
 def question1(path):  
    data=loadmat(path) # loading data from NumberRecognition.mat file
    traindata_eight = data['imageArrayTraining8']  # selecting training and testing arrays of Eight and Nine
====================================================================
flake8:
assign1.py:18:21: W291 trailing whitespace
assign1.py:19:4: E111 indentation is not a multiple of four
assign1.py:19:8: E225 missing whitespace around operator
assign1.py:19:22: E261 at least two spaces before inline comment
assign1.py:20:4: E111 indentation is not a multiple of four
assign1.py:21:4: E111 indentation is not a multiple of four
assign1.py:22:4: E111 indentation is not a multiple of four
assign1.py:22:18: E225 missing whitespace around operator
assign1.py:23:4: E111 indentation is not a multiple of four
assign1.py:23:17: E225 missing whitespace around operator
assign1.py:24:4: E114 indentation is not a multiple of four (comment)
assign1.py:25:4: E111 indentation is not a multiple of four
assign1.py:25:55: E231 missing whitespace after ','
assign1.py:26:4: E111 indentation is not a multiple of four
assign1.py:26:11: E225 missing whitespace around operator
assign1.py:26:31: E231 missing whitespace after ','
assign1.py:26:33: E231 missing whitespace after ','
assign1.py:26:39: E262 inline comment should start with '# '
assign1.py:27:4: E111 indentation is not a multiple of four
assign1.py:27:17: E225 missing whitespace around operator
assign1.py:27:50: E231 missing whitespace after ','
assign1.py:28:4: E111 indentation is not a multiple of four
assign1.py:29:4: E114 indentation is not a multiple of four (comment)
assign1.py:29:4: E265 block comment should start with '# '
assign1.py:30:4: E111 indentation is not a multiple of four
assign1.py:30:11: E225 missing whitespace around operator
assign1.py:30:57: E231 missing whitespace after ','
assign1.py:31:4: E114 indentation is not a multiple of four (comment)
assign1.py:32:4: E111 indentation is not a multiple of four
assign1.py:32:10: E225 missing whitespace around operator
assign1.py:32:35: E231 missing whitespace after ','
assign1.py:32:49: E231 missing whitespace after ','
assign1.py:32:52: E261 at least two spaces before inline comment
assign1.py:32:53: E262 inline comment should start with '# '
assign1.py:33:4: E111 indentation is not a multiple of four
assign1.py:33:10: E225 missing whitespace around operator
assign1.py:33:29: E231 missing whitespace after ','
assign1.py:33:31: E231 missing whitespace after ','
assign1.py:33:34: E261 at least two spaces before inline comment
assign1.py:33:35: E262 inline comment should start with '# '
assign1.py:34:4: E114 indentation is not a multiple of four (comment)
assign1.py:35:4: E111 indentation is not a multiple of four
assign1.py:35:16: E225 missing whitespace around operator
assign1.py:35:47: E231 missing whitespace after ','
assign1.py:36:4: E114 indentation is not a multiple of four (comment)
assign1.py:36:4: E265 block comment should start with '# '
assign1.py:37:4: E111 indentation is not a multiple of four
assign1.py:37:11: E225 missing whitespace around operator
assign1.py:37:56: E231 missing whitespace after ','
assign1.py:38:4: E114 indentation is not a multiple of four (comment)
assign1.py:38:4: E265 block comment should start with '# '
assign1.py:39:4: E111 indentation is not a multiple of four
assign1.py:39:12: E225 missing whitespace around operator
assign1.py:40:4: E111 indentation is not a multiple of four
assign1.py:40:10: E225 missing whitespace around operator
assign1.py:41:4: E111 indentation is not a multiple of four
assign1.py:41:20: E231 missing whitespace after ','
assign1.py:41:25: E261 at least two spaces before inline comment
assign1.py:42:7: E111 indentation is not a multiple of four
assign1.py:42:56: E251 unexpected spaces around keyword / parameter equals
assign1.py:42:58: E251 unexpected spaces around keyword / parameter equals
assign1.py:43:7: E111 indentation is not a multiple of four
assign1.py:44:7: E111 indentation is not a multiple of four
assign1.py:45:7: E111 indentation is not a multiple of four
assign1.py:45:58: E231 missing whitespace after ','
assign1.py:45:68: E261 at least two spaces before inline comment
assign1.py:45:97: W291 trailing whitespace
assign1.py:46:7: E111 indentation is not a multiple of four
assign1.py:46:34: E225 missing whitespace around operator
assign1.py:46:45: E226 missing whitespace around arithmetic operator
assign1.py:46:50: E261 at least two spaces before inline comment
assign1.py:47:4: E114 indentation is not a multiple of four (comment)
assign1.py:47:4: E265 block comment should start with '# '
assign1.py:48:4: E111 indentation is not a multiple of four
assign1.py:49:4: E111 indentation is not a multiple of four
assign1.py:50:4: E111 indentation is not a multiple of four
assign1.py:51:4: E111 indentation is not a multiple of four
assign1.py:52:4: E111 indentation is not a multiple of four
assign1.py:53:4: E111 indentation is not a multiple of four
assign1.py:57:4: E114 indentation is not a multiple of four (comment)
assign1.py:58:4: E111 indentation is not a multiple of four
assign1.py:58:8: E225 missing whitespace around operator
assign1.py:59:4: E114 indentation is not a multiple of four (comment)
assign1.py:59:4: E265 block comment should start with '# '
assign1.py:60:4: E111 indentation is not a multiple of four
assign1.py:61:4: E111 indentation is not a multiple of four
assign1.py:62:4: E114 indentation is not a multiple of four (comment)
assign1.py:62:4: E265 block comment should start with '# '
assign1.py:63:4: E111 indentation is not a multiple of four
assign1.py:63:12: E225 missing whitespace around operator
assign1.py:63:33: E231 missing whitespace after ','
assign1.py:64:4: E111 indentation is not a multiple of four
assign1.py:65:4: E114 indentation is not a multiple of four (comment)
assign1.py:65:4: E265 block comment should start with '# '
assign1.py:65:50: W291 trailing whitespace
assign1.py:66:4: E111 indentation is not a multiple of four
assign1.py:66:19: E225 missing whitespace around operator
assign1.py:67:4: E114 indentation is not a multiple of four (comment)
assign1.py:67:4: E265 block comment should start with '# '
assign1.py:68:4: E111 indentation is not a multiple of four
assign1.py:68:5: E225 missing whitespace around operator
assign1.py:69:4: E114 indentation is not a multiple of four (comment)
assign1.py:69:4: E265 block comment should start with '# '
assign1.py:70:4: E111 indentation is not a multiple of four
assign1.py:70:5: E225 missing whitespace around operator
assign1.py:70:30: E231 missing whitespace after ','
assign1.py:71:4: E114 indentation is not a multiple of four (comment)
assign1.py:71:4: E265 block comment should start with '# '
assign1.py:72:4: E111 indentation is not a multiple of four
assign1.py:72:5: E225 missing whitespace around operator
assign1.py:73:4: E114 indentation is not a multiple of four (comment)
assign1.py:73:4: E265 block comment should start with '# '
assign1.py:74:4: E111 indentation is not a multiple of four
assign1.py:75:4: E111 indentation is not a multiple of four
assign1.py:76:7: E111 indentation is not a multiple of four
assign1.py:76:19: E225 missing whitespace around operator
assign1.py:77:7: E111 indentation is not a multiple of four
assign1.py:77:56: E231 missing whitespace after ','
assign1.py:78:7: E111 indentation is not a multiple of four
assign1.py:78:17: E225 missing whitespace around operator
assign1.py:78:45: E231 missing whitespace after ','
assign1.py:79:4: E111 indentation is not a multiple of four
assign1.py:79:26: W291 trailing whitespace
assign1.py:84:4: E111 indentation is not a multiple of four
assign1.py:84:8: E225 missing whitespace around operator
assign1.py:85:4: E114 indentation is not a multiple of four (comment)
assign1.py:85:4: E265 block comment should start with '# '
assign1.py:86:4: E111 indentation is not a multiple of four
assign1.py:87:4: E111 indentation is not a multiple of four
assign1.py:88:4: E114 indentation is not a multiple of four (comment)
assign1.py:88:4: E265 block comment should start with '# '
assign1.py:89:4: E111 indentation is not a multiple of four
assign1.py:89:12: E225 missing whitespace around operator
assign1.py:89:33: E231 missing whitespace after ','
assign1.py:90:4: E111 indentation is not a multiple of four
assign1.py:91:4: E114 indentation is not a multiple of four (comment)
assign1.py:91:4: E265 block comment should start with '# '
assign1.py:91:50: W291 trailing whitespace
assign1.py:92:4: E114 indentation is not a multiple of four (comment)
assign1.py:92:4: E265 block comment should start with '# '
assign1.py:93:4: E111 indentation is not a multiple of four
assign1.py:93:19: E225 missing whitespace around operator
assign1.py:94:4: E114 indentation is not a multiple of four (comment)
assign1.py:94:4: E265 block comment should start with '# '
assign1.py:95:4: E111 indentation is not a multiple of four
assign1.py:95:5: E225 missing whitespace around operator
assign1.py:96:4: E114 indentation is not a multiple of four (comment)
assign1.py:96:4: E265 block comment should start with '# '
assign1.py:97:4: E111 indentation is not a multiple of four
assign1.py:97:5: E225 missing whitespace around operator
assign1.py:97:30: E231 missing whitespace after ','
assign1.py:98:4: E114 indentation is not a multiple of four (comment)
assign1.py:98:4: E265 block comment should start with '# '
assign1.py:99:4: E111 indentation is not a multiple of four
assign1.py:99:5: E225 missing whitespace around operator
assign1.py:100:4: E114 indentation is not a multiple of four (comment)
assign1.py:100:4: E265 block comment should start with '# '
assign1.py:101:4: E111 indentation is not a multiple of four
assign1.py:101:56: E201 whitespace after '('
assign1.py:102:4: E114 indentation is not a multiple of four (comment)
assign1.py:102:4: E265 block comment should start with '# '
assign1.py:103:4: E111 indentation is not a multiple of four
assign1.py:103:14: E225 missing whitespace around operator
assign1.py:104:4: E111 indentation is not a multiple of four
assign1.py:104:9: E225 missing whitespace around operator
assign1.py:105:4: E111 indentation is not a multiple of four
assign1.py:105:20: E231 missing whitespace after ','
assign1.py:106:7: E111 indentation is not a multiple of four
assign1.py:106:56: E251 unexpected spaces around keyword / parameter equals
assign1.py:106:58: E251 unexpected spaces around keyword / parameter equals
assign1.py:107:7: E111 indentation is not a multiple of four
assign1.py:107:33: E231 missing whitespace after ','
assign1.py:108:7: E111 indentation is not a multiple of four
assign1.py:109:7: E111 indentation is not a multiple of four
assign1.py:109:56: E201 whitespace after '('
assign1.py:109:61: E231 missing whitespace after ','
assign1.py:109:70: E261 at least two spaces before inline comment
assign1.py:110:7: E111 indentation is not a multiple of four
assign1.py:110:33: E225 missing whitespace around operator
assign1.py:110:44: E261 at least two spaces before inline comment
assign1.py:111:4: E114 indentation is not a multiple of four (comment)
assign1.py:111:4: E265 block comment should start with '# '
assign1.py:112:4: E111 indentation is not a multiple of four
assign1.py:113:4: E111 indentation is not a multiple of four
assign1.py:113:78: E231 missing whitespace after ','
assign1.py:114:4: E111 indentation is not a multiple of four
assign1.py:115:4: E111 indentation is not a multiple of four
assign1.py:116:4: E111 indentation is not a multiple of four
assign1.py:117:4: E111 indentation is not a multiple of four
assign1.py:118:1: E305 expected 2 blank lines after class or function definition, found 0
assign1.py:119:4: E111 indentation is not a multiple of four
assign1.py:120:4: E111 indentation is not a multiple of four
assign1.py:120:70: W291 trailing whitespace
assign1.py:121:4: E111 indentation is not a multiple of four
assign1.py:121:70: W292 no newline at end of file
====================================================================
black:
--- assign1.py	2021-10-14 19:02:07.412711 +0000
+++ assign1.py	2021-10-14 19:02:19.301844 +0000
@@ -13,109 +13,113 @@
 import matplotlib.pyplot as plt
 import sklearn
 from sklearn.neighbors import KNeighborsClassifier
 from sklearn.model_selection import train_test_split
 from sklearn.preprocessing import LabelEncoder
-def question1(path):  
-   data=loadmat(path) # loading data from NumberRecognition.mat file
-   traindata_eight = data['imageArrayTraining8']  # selecting training and testing arrays of Eight and Nine
-   traindata_nine = data['imageArrayTraining9']
-   testdata_eight=data['imageArrayTesting8']
-   testdata_nine=data['imageArrayTesting9']
-   # Appending eight and nine arrays
-   X_train = np.append(traindata_eight, traindata_nine,2)  # (28,28,1500)
-   X_train=X_train.transpose(2,0,1)   #(1500,28,28)
-   X_trainingset=X_train.reshape(X_train.shape[0],np.prod(X_train.shape[1:]))
-   X_trainingset.shape
-   #creating training labels
-   X_label=np.append(np.zeros(traindata_eight.shape[-1]),np.ones(traindata_nine.shape[-1]))
-   # Appending testing arrays of Eight and Nine
-   X_test=np.append(testdata_eight,testdata_nine,2) #(28,28,500)
-   X_test=X_test.transpose(2,0,1) #(500,28,28)
-   # Reshaping the testing set
-   Y_testingset=X_test.reshape(X_test.shape[0],np.prod(X_test.shape[1:]))
-   #creating testing labels
-   Y_label=np.append(np.zeros(testdata_eight.shape[-1]),np.ones(testdata_nine.shape[-1]))
-   #Finding error rates using kNN Classifier
-   accuracy=[]
-   errors=[]
-   for k in range(1,21): # for k=20
-      Knn_classifier = KNeighborsClassifier(n_neighbors = k)
-      Knn_classifier.fit(X_trainingset, X_label)
-      pred = Knn_classifier.predict(Y_testingset)
-      accuracy.append(sklearn.metrics.accuracy_score(pred,Y_label)) # Calculating the accuracies 
-      errors.append((np.mean(pred!=Y_label))*100) # Calculating error rate
-   #plotting graph for the error rates
-   plt.figure(figsize=(8, 6))
-   plt.plot(range(1, 21), errors, color='grey', linestyle='dashed', marker='o', markerfacecolor='grey', markersize=8)
-   plt.title('Error Rate K Value')
-   plt.xlabel('K Value')
-   plt.ylabel('Error')
-   plt.show()
+
+
+def question1(path):
+    data = loadmat(path)  # loading data from NumberRecognition.mat file
+    traindata_eight = data["imageArrayTraining8"]  # selecting training and testing arrays of Eight and Nine
+    traindata_nine = data["imageArrayTraining9"]
+    testdata_eight = data["imageArrayTesting8"]
+    testdata_nine = data["imageArrayTesting9"]
+    # Appending eight and nine arrays
+    X_train = np.append(traindata_eight, traindata_nine, 2)  # (28,28,1500)
+    X_train = X_train.transpose(2, 0, 1)  # (1500,28,28)
+    X_trainingset = X_train.reshape(X_train.shape[0], np.prod(X_train.shape[1:]))
+    X_trainingset.shape
+    # creating training labels
+    X_label = np.append(np.zeros(traindata_eight.shape[-1]), np.ones(traindata_nine.shape[-1]))
+    # Appending testing arrays of Eight and Nine
+    X_test = np.append(testdata_eight, testdata_nine, 2)  # (28,28,500)
+    X_test = X_test.transpose(2, 0, 1)  # (500,28,28)
+    # Reshaping the testing set
+    Y_testingset = X_test.reshape(X_test.shape[0], np.prod(X_test.shape[1:]))
+    # creating testing labels
+    Y_label = np.append(np.zeros(testdata_eight.shape[-1]), np.ones(testdata_nine.shape[-1]))
+    # Finding error rates using kNN Classifier
+    accuracy = []
+    errors = []
+    for k in range(1, 21):  # for k=20
+        Knn_classifier = KNeighborsClassifier(n_neighbors=k)
+        Knn_classifier.fit(X_trainingset, X_label)
+        pred = Knn_classifier.predict(Y_testingset)
+        accuracy.append(sklearn.metrics.accuracy_score(pred, Y_label))  # Calculating the accuracies
+        errors.append((np.mean(pred != Y_label)) * 100)  # Calculating error rate
+    # plotting graph for the error rates
+    plt.figure(figsize=(8, 6))
+    plt.plot(range(1, 21), errors, color="grey", linestyle="dashed", marker="o", markerfacecolor="grey", markersize=8)
+    plt.title("Error Rate K Value")
+    plt.xlabel("K Value")
+    plt.ylabel("Error")
+    plt.show()
 
 
 def question2(path):
-   # Reading data from placement_Data_Full_Class.csv file
-   data=pd.read_csv(path)
-   #Checking for any missing values
-   print(data.isnull().sum())
-   print(data.shape)
-   #Dropping the column which has missing values(Salary column)
-   new_data=data.drop(['salary'],axis=1)
-   print(new_data.shape)
-   #Encoding Categorical data of the label column 
-   new_data.status=LabelEncoder().fit_transform(new_data.status)
-   #Storing label values in Y
-   Y=new_data.status
-   #Storing all the columns except for labels in X
-   X=new_data.drop(['status'],axis=1)
-   #Converting categorical data using dummies
-   X=pd.get_dummies(X)
-   #calculating AUC value for each feature
-   auc_values = []
-   for i in X.columns:
-      features_auc= np.array(X[i])
-      auc_values.append(sklearn.metrics.roc_auc_score(Y,features_auc))
-      sorted_auc=sorted(np.array(auc_values),reverse=True)
-   print(sorted_auc[:10])   
+    # Reading data from placement_Data_Full_Class.csv file
+    data = pd.read_csv(path)
+    # Checking for any missing values
+    print(data.isnull().sum())
+    print(data.shape)
+    # Dropping the column which has missing values(Salary column)
+    new_data = data.drop(["salary"], axis=1)
+    print(new_data.shape)
+    # Encoding Categorical data of the label column
+    new_data.status = LabelEncoder().fit_transform(new_data.status)
+    # Storing label values in Y
+    Y = new_data.status
+    # Storing all the columns except for labels in X
+    X = new_data.drop(["status"], axis=1)
+    # Converting categorical data using dummies
+    X = pd.get_dummies(X)
+    # calculating AUC value for each feature
+    auc_values = []
+    for i in X.columns:
+        features_auc = np.array(X[i])
+        auc_values.append(sklearn.metrics.roc_auc_score(Y, features_auc))
+        sorted_auc = sorted(np.array(auc_values), reverse=True)
+    print(sorted_auc[:10])
 
 
 def question3(path):
     # Reading data from placement_Data_Full_Class.csv file
-   data=pd.read_csv(path)
-   #Checking for any missing values
-   print(data.isnull().sum())
-   print(data.shape)
-   #Dropping the column which has missing values(Salary column)
-   new_data=data.drop(['salary'],axis=1)
-   print(new_data.shape)
-   #Encoding Categorical data of the label column 
-   #labelencoding=LabelEncoder()
-   new_data.status=LabelEncoder().fit_transform(new_data.status)
-   #Storing label values in Y
-   Y=new_data.status
-   #Storing all the columns except for labels in X
-   X=new_data.drop(['status'],axis=1)
-   #Converting categorical data using dummies
-   X=pd.get_dummies(X)
-   #Splitting the data into testing and training sets
-   X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.2, random_state=42)
-   #finding error rate using kNN classifier
-   accuracies=[]
-   error=[]
-   for k in range(1,21):
-      knn_classifier = KNeighborsClassifier(n_neighbors = k)
-      knn_classifier.fit(X_train,y_train)
-      pred = knn_classifier.predict(X_test)
-      accuracies.append(sklearn.metrics.accuracy_score( pred,y_test)) # Calculating accuracies
-      error.append((np.mean(pred!=y_test))) # Calculating error rates
-   #plotting the graph for error rates
-   plt.figure(figsize=(8, 6))
-   plt.plot(range(1, 21), error, color='grey', linestyle='dashed', marker='o',markerfacecolor='grey', markersize=8)
-   plt.title('Error Rate K Value')
-   plt.xlabel('K Value')
-   plt.ylabel('Error')
-   plt.show()
+    data = pd.read_csv(path)
+    # Checking for any missing values
+    print(data.isnull().sum())
+    print(data.shape)
+    # Dropping the column which has missing values(Salary column)
+    new_data = data.drop(["salary"], axis=1)
+    print(new_data.shape)
+    # Encoding Categorical data of the label column
+    # labelencoding=LabelEncoder()
+    new_data.status = LabelEncoder().fit_transform(new_data.status)
+    # Storing label values in Y
+    Y = new_data.status
+    # Storing all the columns except for labels in X
+    X = new_data.drop(["status"], axis=1)
+    # Converting categorical data using dummies
+    X = pd.get_dummies(X)
+    # Splitting the data into testing and training sets
+    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
+    # finding error rate using kNN classifier
+    accuracies = []
+    error = []
+    for k in range(1, 21):
+        knn_classifier = KNeighborsClassifier(n_neighbors=k)
+        knn_classifier.fit(X_train, y_train)
+        pred = knn_classifier.predict(X_test)
+        accuracies.append(sklearn.metrics.accuracy_score(pred, y_test))  # Calculating accuracies
+        error.append((np.mean(pred != y_test)))  # Calculating error rates
+    # plotting the graph for error rates
+    plt.figure(figsize=(8, 6))
+    plt.plot(range(1, 21), error, color="grey", linestyle="dashed", marker="o", markerfacecolor="grey", markersize=8)
+    plt.title("Error Rate K Value")
+    plt.xlabel("K Value")
+    plt.ylabel("Error")
+    plt.show()
+
+
 if __name__ == "__main__":
-   question1(r"C:\Users\Nikhila reddy\NumberRecognition.mat")
-   question2(r"C:\Users\Nikhila reddy\Placement_Data_Full_Class.csv") 
-   question3(r"C:\Users\Nikhila reddy\Placement_Data_Full_Class.csv")
\ No newline at end of file
+    question1(r"C:\Users\Nikhila reddy\NumberRecognition.mat")
+    question2(r"C:\Users\Nikhila reddy\Placement_Data_Full_Class.csv")
+    question3(r"C:\Users\Nikhila reddy\Placement_Data_Full_Class.csv")
====================================================================
-->
