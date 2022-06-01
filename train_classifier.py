import cv2
import os
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
import keras 
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPool2D, Dropout, Flatten
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt

def load_data(data_path):
    # load monkbrill jpg data
    data = dict()
    data['label'] = []
    data['filename'] = []
    data['data'] = []   

    for subdir in os.listdir(data_path):
        current_path = os.path.join(data_path, subdir)

        for file in os.listdir(current_path):
            im = cv2.imread(os.path.join(current_path, file), cv2.IMREAD_GRAYSCALE)
            im = cv2.resize(im, (38, 48)) # (width, height)
            data['label'].append(subdir)
            data['filename'].append(file)
            data['data'].append(im)

    return data

def build_model():
    # Build LeNet model with drop out
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(48, 38, 1)))
    model.add(MaxPool2D(strides=2))
    model.add(Conv2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))
    model.add(MaxPool2D(strides=2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(27, activation='softmax'))

    return model


if __name__ == '__main__':

    # load data
    data_path = "monkbrill-jpg/"
    data = load_data(data_path)

    # one-hot encode class labels
    encoder = LabelBinarizer()
    onehot_labels = encoder.fit_transform(data['label'])

    # split in training and validation data and put in np array
    X_train, X_val, y_train, y_val = train_test_split(data['data'], onehot_labels, test_size=0.2, random_state=0, shuffle=True)
    X_train = np.array(X_train) 
    y_train = np.array(y_train) 
    X_val = np.array(X_val) 
    y_val = np.array(y_val)

    # instantiate model
    model = build_model()

    # Compile model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    # create directory to save trained model
    if not os.path.exists('models'):
        os.makedirs('models')
    model_filepath = 'models/trained_LeNet_model.h5'

    # set callback to save model when validation loss is at minimum
    callback = ModelCheckpoint(filepath=model_filepath, save_weights_only=True, monitor='val_loss', mode='min', save_best_only=True)

    # train model
    history = model.fit(X_train, y_train, epochs=10, validation_data = (X_val, y_val), callbacks = [callback], verbose=1)

    # Plot the loss of model
    plt.plot(history.history['loss'], label = 'Training loss')
    plt.plot(history.history['val_loss'], label = 'Validation loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot the accuracy of model
    plt.plot(history.history['accuracy'], label = 'Training accuracy')
    plt.plot(history.history['val_accuracy'], label = 'Validation accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid()
    plt.show()

