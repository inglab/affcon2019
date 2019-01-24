from keras.layers import Input, Dense, LeakyReLU, Dropout, BatchNormalization
from keras.models import Model, load_model
from numpy import genfromtxt
import numpy as np
import csv

def load_unlabeled(label_type):
    my_data = genfromtxt('./processed/unlabeled_train.csv', delimiter=',')
    X=my_data[:,:]
    return X

def myDistance(u, v):
    distance = 0.0
    u = u[0]
    v = v[0]
    for idx in range(u.shape[0]):
    #print(u.shape)
    #for idx in range(concept_len):
        distance += abs(u[idx]-v[idx])
    return distance

def myMse(u,v):
    return ((u - v)**2).mean(axis=None)

def mycross_entropy(predictions, targets):
    N = 512
    ce = -np.sum(targets*np.log(predictions))/N
    return ce

def labeling(model,X_test,label_type,train_val):
    total=0
    total_loss=[]
    true_val=0
    yesLoss=[]
    noLoss=[]
    for idx in range(len(X_test)):
        total+=1
        input_test = X_test[idx].reshape((1,512))
        predict_test = model.predict(input_test)
        loss = myDistance(input_test, predict_test)
        #loss = myMse(input_test, predict_test)
        #loss = mycross_entropy( predict_test,input_test)
        total_loss.append(loss)

    pct=3
    k = (int)(0.01*pct*total)
    print(k)
    idxs = np.argpartition(total_loss, k)[:k]
    p_file=open('./picked/label_'+str(label_type)+'_'+str(train_val)+'.csv','w')
    wr = csv.writer(p_file, dialect='excel')
    for i in idxs:
        wr.writerow(X_test[i])
    p_file.close()

def my_run(label_type,train_val):
    #label_type 0 for agency 1 for social
    #train_val 0 when training with no, 1 when training with yes
    model = load_model('./model'+str(label_type)+'_'+str(train_val)+'.h5')
    X_test=load_unlabeled(label_type)
    labeling(model,X_test,label_type,train_val)
my_run(1,1)
my_run(1,0)
my_run(0,1)
my_run(0,0)
