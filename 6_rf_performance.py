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
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

def my_rf(un,samples,label_type):
    #word_size=3
    #k=5
    print(samples)
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
    X=my_data[:,:]
    y_social=my_label[:]
    #X=my_data[:,:999]
    #X=my_data[:,:-2]
    X_test=test_d[:,:]
    y_test=test_l[:]
    clf = RandomForestClassifier(n_estimators=samples)
    clf.fit(X, y_social)
    y_pred=clf.predict(X_test)
    print(precision_recall_fscore_support(y_test, y_pred, average=None))
    print(accuracy_score(y_test, y_pred))

my_rf(0,10,1)
my_rf(1,10,1)

#my_rf(0,100)
#my_rf(1,100)
#my_rf(0,1000)
#my_rf(1,1000)
