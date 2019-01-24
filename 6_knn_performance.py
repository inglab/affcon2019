import csv
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split
from numpy import genfromtxt
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

def my_knn(k,un,label_type):
    #word_size=3
    #k=5
    print(k)
    if un==1:
        print("combined")
        d_file = genfromtxt('./picked/combined_'+str(label_type)+'.csv', delimiter=',')
        my_data=d_file[:,:512]
        my_label=d_file[:,512]

    else:
        print("original")
        d_file=genfromtxt('./processed/labeled_train.csv', delimiter=',')
        my_data =d_file[:,:512]
        my_label =d_file[:,512+label_type]

    t_file=genfromtxt('./processed/labeled_test.csv', delimiter=',')
    test_d=t_file[:,:512]
    test_l=t_file[:,512+label_type]
    #X=my_data[:,:-(2+len(concepts))]
    X=my_data[:,:]
    y_social=my_label[:]
    #X=my_data[:,:999]
    #X=my_data[:,:-2]
    X_test=test_d[:,:]
    y_test=test_l[:]

    neigh = KNeighborsClassifier(n_neighbors=k,p=1)
    neigh.fit(X, y_social)
    y_pred=neigh.predict(X_test)
    print(precision_recall_fscore_support(y_test, y_pred, average=None))
    print(accuracy_score(y_test, y_pred))

#social
k=5
#my_knn(k,0,1)
#my_knn(k,1,1)

my_knn(k,0,0)
#my_knn(k,1,0)
