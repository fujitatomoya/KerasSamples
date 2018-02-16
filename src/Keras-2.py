import numpy as np
from keras.layers import Input, Dense
from keras.models import Model

data = np.random.rand(250,2)
labels = (np.sum(data, axis=1) > 0.5) * 2 - 1

input = Input(shape=(2,))
output = Dense(1, activation='tanh')

model = Model(input=input, output=output(input))
model.compile('adam', 'hinge', metrics=['accuracy'])

output.set_weights([np.array([[1.0], [1.0]]), np.array([-0.5])])

test = np.random.rand(200, 2)
predict = np.sign(model.predict(test).flatten())
real = (np.sum(test, axis=1) > 0.5) * 2 - 1
print(sum(predict == real) / 200.0)
