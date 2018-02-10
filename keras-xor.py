# -*- coding: utf-8 -*-
"""
Source : https://gist.github.com/stewartpark/187895beb89f0a1b3a54
"""

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np 

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

model = Sequential()
model.add(Dense(2, input_dim=2, activation='tanh'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1))

model.fit(X, y, batch_size=1, nb_epoch=500)
print(model.predict_proba(X))

for layer in model.layers:
    weights = layer.get_weights()
    print(weights)
    
model.save_weights("model.h5")