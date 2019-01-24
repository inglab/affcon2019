from keras.layers import Input, Dense, LeakyReLU, Dropout, BatchNormalization
from keras.models import Model, load_model
from numpy import genfromtxt
import numpy as np

def autoencoder_model():

    input_log = Input(shape=(512,))

    encoded = Dense(256, kernel_initializer='glorot_normal', activation='linear')(input_log)
    #encoded = Dropout(0.5)(encoded)
    #encoded= BatchNormalization()(encoded)
    encoded = LeakyReLU(alpha=0.01)(encoded)

    encoded = Dense(128, kernel_initializer='glorot_normal', activation='linear')(encoded)
   # encoded = Dropout(0.5)(encoded)
    #encoded= BatchNormalization()(encoded)
    encoded = LeakyReLU(alpha=0.01)(encoded)

    encoded = Dense(64, kernel_initializer='glorot_normal', activation='linear')(encoded)
   # encoded = Dropout(0.5)(encoded)
    #encoded= BatchNormalization()(encoded)
    encoded = LeakyReLU(alpha=0.01)(encoded)

    #decompression (dense -> dropout -> leakyrelu)
    decoded = Dense(128, kernel_initializer='glorot_normal', activation='linear')(encoded)
   # decoded = Dropout(0.5)(decoded)
    #decoded= BatchNormalization()(decoded)
    decoded = LeakyReLU(alpha=0.01)(decoded)
    decoded = Dense(256, kernel_initializer='glorot_normal', activation='linear')(decoded)
   # decoded = Dropout(0.5)(decoded)
    #decoded= BatchNormalization()(decoded)
    decoded = LeakyReLU(alpha=0.01)(decoded)
    #decoded = Dense(512, kernel_initializer='glorot_normal', activation='linear')(decoded)
   # decoded = Dropout(0.5)(decoded)
    #decoded= BatchNormalization()(decoded)
    decoded = LeakyReLU(alpha=0.01)(decoded)
    decoded = Dense(512, kernel_initializer='glorot_normal', activation='sigmoid')(decoded)
    autoencoder = Model(input_log, decoded)

    #print(autoencoder.summary())
    autoencoder.compile(optimizer='sgd', loss='mean_squared_error')
    #autoencoder.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    #autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    return autoencoder

def load_train(label_type,train_val):
    my_data = genfromtxt('./processed/labeled_train.csv', delimiter=',')
    my_data=my_data[np.where(my_data[:,512+label_type]==train_val)]
    print(len(my_data))
    X=my_data[:,:512]
    label=my_data[:,512+label_type]
    return X,label
    #return X,np.concatenate((np.array(X),np.array(y)),axis=1)

def load_test(label_type):
    my_data = genfromtxt('./processed/labeled_test.csv', delimiter=',')
    X=my_data[:,:512]
    label=my_data[:,512+label_type]

    return X,label

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

def measure(model, X_test,y_test,train_val):

    total=0
    total_loss=[]
    true_val=0
    yesLoss=[]
    noLoss=[]
    for idx in range(len(X_test)):
        total+=1
        input_test = X_test[idx].reshape((1,512))
        predict_test = model.predict(input_test)
        t_l=y_test[idx]
        loss = myDistance(input_test, predict_test)
        #loss = myMse(input_test, predict_test)
        #loss = mycross_entropy( predict_test,input_test)

        total_loss.append(loss)
        #print(loss)
        if(t_l == 1) :
            #print(loss)
            yesLoss.append(loss)
        else:
            #print(loss)
            noLoss.append(loss)
    #choose smallest 3% of loss
    pct=3
    k = (int)(0.01*pct*total)
    print(k)
    idxs = np.argpartition(total_loss, k)[:k]
    total2=0
    correct2=0
    for i in idxs:
        total2+=1
        t_l=y_test[i]
        if t_l==train_val:
            correct2+=1
    print("Total of %d percent %d, Correct: %d"%(pct,total2,correct2))
    print("Correct percentage %.2f"%(correct2/total2))

    print("Yes data loss(%d): %f" % (len(yesLoss), np.average(np.array(yesLoss))))
    print("No data loss(%d): %f" % (len(noLoss), np.average(np.array(noLoss))))

def my_run(label_type,train_val):
    #label_type 0 for agency 1 for social
    #train_val 0 when training with no, 1 when training with yes
    X_train,y_train=load_train(label_type,train_val)
    model = autoencoder_model()
    model.fit(X_train, X_train,
                    epochs=30,
                    batch_size=32,
                    shuffle=True)
    model.save('model'+str(label_type)+'_'+str(train_val)+'.h5')
    X_test,y_test=load_test(label_type)
    measure(model,X_test,y_test,train_val)
    
my_run(1,1)
my_run(1,0)
my_run(0,1)
my_run(0,0)
