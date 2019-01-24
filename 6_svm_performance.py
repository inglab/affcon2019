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

def my_svm(kernel,un,label_type):
    #word_size=3
    #k=5
    print(kernel)
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

    X_test=test_d[:,:]
    y_test=test_l[:]
    gamma=0.1
    clf = svm.SVC(gamma=gamma,kernel=kernel)
    clf.fit(X, y_social)
    y_pred=clf.predict(X_test)
    print(precision_recall_fscore_support(y_test, y_pred, average=None))
    print(accuracy_score(y_test, y_pred))

#social
#my_svm('linear',0,1)
#my_svm('linear',1,1)
#my_svm('rbf',0,1)
#my_svm('rbf',1,1)

#agency
my_svm('linear',0,0)
#my_svm('linear',1,0)
#my_svm('rbf',0,0)
#my_svm('rbf',1,0)
