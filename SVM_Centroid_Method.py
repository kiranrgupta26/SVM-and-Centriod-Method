# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 13:20:45 2019

@author: Kiran Rambilas Gupta: 1001726759  , Sumanth : 1001738842
"""
#SubRoutine Task Code

import numpy as np
import math
import sys
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

f = open("HandWrittenLetters.txt", "r")

line = f.readline()
arr = np.array([line.split(",")])
line = f.readline()

while line:    
    newarr = np.array(line.split(","))
    arr = np.append(arr,[newarr],axis=0)
    line = f.readline()

f.close()

f1=open("testDataX.txt", "r")
line1 = f1.readline()
tarr1 = np.array([line1.split(",")])
line1 = f1.readline()

while line1:    
    testarr = np.array(line1.split(","))
    tarr1 = np.append(tarr1,[testarr],axis=0)
    line1 = f1.readline()

f1.close()

tarr1 = np.array(np.transpose(tarr1)) 


#Subroutine 1
temp_arr = np.array([arr[:,0]])
def pickDataClass(arr1):
    for j in arr1:
        for i in range((arr.shape[1])):
            if int(arr[0][i]) == j:
                value = i               
                temp =np.array(arr[:,value])
                global temp_arr
                temp_arr = np.append(temp_arr,[temp],axis=0)
    temp_arr = np.array(np.delete(temp_arr,0,0))   
    temp_arr = np.array(np.transpose(temp_arr)) 
    global test_data_xy
    test_data_xy = np.array([temp_arr[:,0]])
    global train_data_xy
    train_data_xy = np.array([temp_arr[:,0]])


#pickDataClass([1,2,3,4,5])     

#Subroutine 4  
def letter_To_digit_Convert(data):
    class_data=[]
    for i in range(len(data)):
        class_data.append(ord(data[i].lower())-96)
        #print(ord(data[i].lower())-96)  
    pickDataClass(class_data)
    

#letter_To_digit_Convert("ABCDE")

#Subroutine 2
def splitData2TestTrain(number_per_class,test_instance):
    data = test_instance.split(":")
    k=0
    while(k < temp_arr.shape[1]):
        for i in range(k,int(data[1])+k):
            test_data = np.array(temp_arr[:,i])
            global test_data_xy
            test_data_xy = np.append(test_data_xy,[test_data],axis=0)
    
        for j in range(k+int(data[1]),number_per_class+k):
            train_data = np.array(temp_arr[:,j])
            global train_data_xy
            train_data_xy = np.append(train_data_xy,[train_data],axis=0)
            
        k = k+number_per_class
        
    test_data_xy = np.array(np.delete(test_data_xy,0,0))
    train_data_xy = np.array(np.delete(train_data_xy,0,0))
    
    test_data_xy = test_data_xy.astype(np.int)
    train_data_xy = train_data_xy.astype(np.int)
    
    global y_test
    y_test = np.array(test_data_xy[:,0])
    
    
    global y_train
    y_train = np.array(train_data_xy[:,0])
    
    
#splitData2TestTrain(10,'1:1')


 #Subroutine 3
def write_to_file():
    
    f = open("trainX.txt", "a")
    f.write(str(train_data_xy[:,1:]))
    f.close()

    f = open("trainY.txt", "a")
    f.write(str(y_train))
    f.close()

    f = open("testX.txt", "a")
    f.write(str(test_data_xy[:,1:]))
    f.close()

    f = open("testY.txt", "a")
    f.write(str(y_test))
    f.close()
    
#End OF SubRoutine Task Code    
#-----------------------------------------------------------------------------------------    
#5 Fold Cross Validation
np.set_printoptions(threshold=sys.maxsize)
from sklearn.model_selection import KFold

pickDataClass([1,2,3,4,5])  
splitData2TestTrain(39,'1:10')  

sc2 = StandardScaler()
temp_arr1 = np.array(np.transpose(temp_arr)) 
centroid_data1 = np.array([test_data_xy[0,:]])
def k_fold_validation():
    kfold = KFold(5, True, 1)  # 5 fold validation. 
    for train, test in kfold.split(temp_arr1):   
        global temp_temp
        temp_temp = temp_arr1[train]
        temp_test = temp_arr1[test]
        temp_temp=temp_temp.astype(np.int)
        temp_test=temp_test.astype(np.int)
        
        temp_temp[:,1:] = sc2.fit_transform(temp_temp[:,1:])
        temp_test[:,1:] = sc2.transform(temp_test[:,1:])
        SVM(temp_temp,temp_test)
        KNN(temp_temp,temp_test)
        
        centroid_method1()
        classify_centroid_method1(temp_test)       
    global centroid_data1
    centroid_data1 = np.array(np.delete(centroid_data1,0,0))
      

svm=[]
def SVM(x_temp_train,x_temp_test):   
    classifier = SVC(kernel = 'rbf', random_state = 0)   
    classifier.fit(x_temp_train[:,1:], x_temp_train[:,0])
    global y_pred  
    y_pred = classifier.predict(x_temp_test[:,1:])
    svm.append(accuracy_score(x_temp_test[:,0], y_pred))
    

knn=[]
def KNN(x_temp_train,x_temp_test):
    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2,weights='distance') # Using Distance Weighted Voting
    classifier.fit(x_temp_train[:,1:], x_temp_train[:,0])
    y_pred = classifier.predict(x_temp_test[:,1:])
    knn.append(accuracy_score(x_temp_test[:,0], y_pred))
    

cm=[]
def centroid_method1():
    for i in np.unique(temp_temp[:,0]):
        global class_data_instance
        class_data_instance=temp_temp[temp_temp[:,0]==i]
        class_data_instance=class_data_instance.astype(np.float)
        class_instance = len(temp_temp[temp_temp[:,0]==i])
        global value1
        value1 = np.sum(class_data_instance[:,:],axis=0)
        value1 = np.divide(value1,class_instance)
        global centroid_data1
        centroid_data1 = np.append(centroid_data1,[value1],axis=0)
 
y_pred_centroid=[]       
def classify_centroid_method1(test_data):
    test_data=test_data.astype(np.int)
    global centroid_data1
    centroid_data1=centroid_data1.astype(np.float)
    min_distance1=9999999
    centroid_distance1=0
    class_label1=0
    global y_pred_centroid
    y_pred_centroid=[]
    for k in range(test_data.shape[0]):     
        min_distance1=9999999
        for i in np.unique(test_data[:,0]):
            centroid_distance1=0
            for j in range(test_data.shape[1]):
                centroid_distance1 = centroid_distance1+math.pow((centroid_data1[i][j] - test_data[k][j]),2)
            centroid_distance1 = math.sqrt(centroid_distance1)
                
            if min_distance1 > centroid_distance1:
                min_distance1 = centroid_distance1
                class_label1 = i
                
        y_pred_centroid.append(class_label1)
    cm.append(accuracy_score(test_data[:,0], y_pred_centroid))    
    
    
k_fold_validation()
print('SVM Accuracy for K-Fold Validation ',svm)
print('KNN Accuracy for K-Fold Validation ',knn)
print('Centroid Method Accuracy for K-Fold Validation ',cm)
  
plt.plot(knn)
plt.title("KNN")
plt.ylabel('Accuracy')
plt.xlabel('Training Sample')
plt.show()
    
plt.plot(svm)
plt.title("SVM")
plt.ylabel('Accuracy')
plt.xlabel('Training Sample')
plt.show()
    
plt.plot(cm)
plt.title("Centroid Method")
plt.ylabel('Accuracy')
plt.xlabel('Training Sample')
plt.show()
    
        