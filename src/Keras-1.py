import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation

data = np.random.rand(1000,5)
labels1 = (np.sum(data, axis=1) > 2.5) * 1
labels2 = (np.sum(data, axis=1) > 2.5) * 2 - 1
# to make the one hot output
labels1 = np_utils.to_categorical(labels1)

# input 5
# hidden 20 x Activation'ReLU' (rectified linear unit)
# output 2 x Activation'softmax'
model1 = Sequential()
model2 = Sequential()

'''
Activation
  softsign   : -1 to 1, smooth carve.
  tanh       : -1 to 1, sharp carve.
  sigmoid    : 0 to 1, intput(0) -> output(0.5)
  softplus   : => 0
  relu       : => 0
  linear     : ...
  softmax    : ...
'''
model1.add(Dense(20, input_dim=5, activation='relu'))
model1.add(Dense(2, activation='softmax'))
model2.add(Dense(20, input_dim=5, activation='tanh'))
model2.add(Dense(1, activation='tanh'))

'''
Optimizer
 sgd, rsmprop, adagrad, adadelta, adam, adamax, nadam...
 https://qiita.com/tokkuman/items/1944c00415d129ca0ee9
Objective
 mse, msa, mspa, msle, hinge, squared_hinge, binary_crossentropy, ...
 https://keras.io/ja/objectives/
'''
# compile the model with label
model1.compile('rmsprop', 'categorical_crossentropy', metrics=['accuracy'])
model2.compile('adam', 'hinge', metrics=['accuracy'])
model1.fit(data, labels1, nb_epoch=300, validation_split=0.2)
model2.fit(data, labels2, nb_epoch=150, validation_split=0.2)

# test the neural network
test = np.random.rand(200, 5)
predict1 = np.argmax(model1.predict(test), axis=1)
predict2 = np.sign(model2.predict(test).flatten())
real1 = (np.sum(test, axis=1) > 2.5) * 1
real2 = (np.sum(test, axis=1) > 2.5) * 2 - 1
print(sum(predict1 == real1) / 200.0)
print(sum(predict2 == real2) / 200.0)
