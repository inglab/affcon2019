import numpy as np
from numpy import genfromtxt
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

def my_xg(un,label_type):
    #word_size=3
    #k=5
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

    model = XGBClassifier()
    model.fit(X, y_social)
    y_pred=model.predict(X_test)
    print(precision_recall_fscore_support(y_test, y_pred, average=None))
    print(accuracy_score(y_test, y_pred))

#social
#my_xg(0,1)
#my_xg(1,1)
my_xg(0,0)
#my_xg(1,0)
