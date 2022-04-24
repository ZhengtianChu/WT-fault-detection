from keras import models
from keras import layers
import keras
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from keras.wrappers import scikit_learn
from sklearn.ensemble import AdaBoostClassifier
from sklearn import decomposition
from sklearn import svm
import utils

def createANN(no_neuron, input_shape):
     # build model
    model = models.Sequential()
    model.add(layers.Dense(no_neuron,activation='relu', input_shape=(input_shape,)))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1,activation='sigmoid'))
    model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['accuracy']) 
#     print(model.summary())
    return model

def KFoldCV(model, X, Y, k, batch_size, epochs):
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=0)
        ]
    
    kf = KFold(n_splits=k,shuffle=False)
    acc_per_fold = []
    loss_per_fold = []
    fold_no = 1
    for train_index,test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        print("[INFO] training model...")
        history = model.fit(x=X_train, y=Y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks)
        scores = model.evaluate(X_test, Y_test, verbose=0)

        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])
        fold_no = fold_no + 1
        
    print('------------------------------------------------------------------------')
    print('[INFO]Average scores for batchsize:'+ str(batch_size))
    print(f'[INFO] Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'[INFO] Loss: {np.mean(loss_per_fold)}')
    print('------------------------------------------------------------------------')
    
    return np.mean(acc_per_fold)

def ANNModel(X_train, Y_train):
    no_neuron = [30,40,50,60]
    batch_size = [16, 32, 64]
#     no_neuron = [60]
#     batch_size = [32]
    
    best_acc = 0
    opt_bs = 16
    opt_non = 30
    best_model = createANN(60, X_train.shape[1])
    
    for non in no_neuron:
        model = createANN(non, X_train.shape[1])
        print("-------------------------------------------")
        print("[INFO] no of neurons: " + str(non))
        for bsize in batch_size:
            acc = KFoldCV(model, X_train, Y_train, 5, bsize, 50)
            if acc > best_acc:
                best_acc = acc
                opt_bs = bsize
                opt_non = non
                best_model = model
                
    best_model.save("myANNModel")
                
    print("[INFO] Best acc:" + str(best_acc) + "best bs:" + str(opt_bs) +  ",best non" + str(opt_non))
    
    return best_model


def AdaBoost(X, Y):
    clf = AdaBoostClassifier(n_estimators=100, random_state=24, learning_rate = 0.1)
    clf.fit(X, Y)
    return clf
    
    
      
        
        
    

