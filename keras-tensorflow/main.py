#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


if __name__ == '__main__':
    iris_data = load_iris() # load the iris dataset
    print('Example data: ')
    print(iris_data.data[:5])
    print('Example labels: ')
    print(iris_data.target[:5])

    x = iris_data.data
    y_ = iris_data.target.reshape(-1, 1) # Convert data to a single column
    encoder = OneHotEncoder(sparse=False)
    y = encoder.fit_transform(y_)
    #print(y)

    # Split the data for training and testing
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20)

    # Build the model
    model = Sequential()
    model.add(Dense(10, input_shape=(4,), activation='relu', name='fc1'))
    model.add(Dense(10, activation='relu', name='fc2'))
    model.add(Dense(3, activation='softmax', name='output'))

    # Adam optimizer with learning rate of 0.001
    optimizer = Adam(lr=0.001)
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    print('Neural Network Model Summary: ')
    print(model.summary())

    # Train the model
    model.fit(train_x, train_y, verbose=2, batch_size=5, epochs=200)
    # Save the model in h5 format 
    model.save("/root/demo_nfs/keras_iris_test.h5")

    model_json = model.to_json()  # save just the config. replace with "to_yaml" for YAML serialization
    with open("/root/demo_nfs/keras_iris_test.json", "w") as f:
        f.write(model_json)

    model.save_weights('/root/demo_nfs/keras_iris_test_weight.h5') # save just the weights.


