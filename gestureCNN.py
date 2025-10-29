#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Masonwabe Shaun Nkombisa 
"""

# UPDATED IMPORTS FOR TENSORFLOW 2.x / KERAS 3.x
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.optimizers import SGD, RMSprop, Adam  # Changed: adam -> Adam
from tensorflow.keras.utils import to_categorical  # Changed: np_utils -> to_categorical

from tensorflow.keras import backend as K

# Set image data format
K.set_image_data_format('channels_first')
	
import numpy as np
import os

from PIL import Image
# SKLEARN
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import json

import cv2
import matplotlib
from matplotlib import pyplot as plt

# input image dimensions
img_rows, img_cols = 200, 200

# number of channels
img_channels = 1

# Batch_size to train
batch_size = 32

## Number of output classes
nb_classes = 5

# Number of epochs to train
nb_epoch = 100

# Total number of convolutional filters to use
nb_filters = 32
# Max pooling
nb_pool = 2
# Size of convolution kernel
nb_conv = 3

#%%
#  data
path = "./"
path1 = "./gestures"
path2 = './imgfolder_b'

WeightFileName = []

# outputs
output = ["OK", "NOTHING","PEACE", "PUNCH", "STOP"]

jsonarray = {}

#%%
def update(plot):
    global jsonarray
    h = 450
    y = 30
    w = 45
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for items in jsonarray:
        mul = (jsonarray[items]) / 100
        cv2.line(plot,(0,y),(int(h * mul),y),(255,0,0),w)
        cv2.putText(plot,items,(0,y+5), font , 0.7,(0,255,0),2,1)
        y = y + w + 30

    return plot

#%%
def debugme():
    import pdb
    pdb.set_trace()

#%%
def convertToGrayImg(path1, path2):
    listing = os.listdir(path1)
    for file in listing:
        if file.startswith('.'):
            continue
        img = Image.open(path1 +'/' + file)
        grayimg = img.convert('L')
        grayimg.save(path2 + '/' +  file, "PNG")

#%%
def modlistdir(path, pattern = None):
    listing = os.listdir(path)
    retlist = []
    for name in listing:
        if pattern == None:
            if name.startswith('.'):
                continue
            else:
                retlist.append(name)
        elif name.endswith(pattern):
            retlist.append(name)
            
    return retlist


# Load CNN model
def loadCNN(bTraining = False):
    global get_output
    model = Sequential()
    
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv),
                        padding='valid',
                        input_shape=(img_channels, img_rows, img_cols)))
    convout1 = Activation('relu')
    model.add(convout1)
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv)))
    convout2 = Activation('relu')
    model.add(convout2)
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    
    # Model summary
    model.summary()
    # Model config details
    model.get_config()
    
    if not bTraining:
        WeightFileName = modlistdir('./models/','.h5')
        if len(WeightFileName) == 0:
            print('Error: No pretrained weight file found.')
            return 0
        else:
            print('Found these weight files - {}'.format(WeightFileName))
        w = int(input("Which weight file to load (enter the INDEX, starting from 0): "))
        fname = './models/' + WeightFileName[int(w)]
        print("loading ", fname)
        model.load_weights(fname)

    # Get layer output for predictions
    # layer = model.layers[-1]
    # get_output = K.function([model.layers[0].input], [layer.output])
    
    return model

# This function does the guessing work based on input images
def guessGesture(model, img):
    global output, get_output, jsonarray
    
    # Load image and flatten it
    image = np.array(img).flatten()
    
    # reshape it
    image = image.reshape(img_channels, img_rows, img_cols)
    
    # float32
    image = image.astype('float32') 
    
    # normalize it
    image = image / 255
    
    # reshape for NN
    rimage = image.reshape(1, img_channels, img_rows, img_cols)
    
    # Get predictions
    prob_array = model.predict(rimage, verbose=0)
    
    d = {}
    i = 0
    for items in output:
        d[items] = prob_array[0][i] * 100
        i += 1
    
    # Get the output with maximum probability
    import operator
    
    guess = max(d.items(), key=operator.itemgetter(1))[0]
    prob  = d[guess]

    if prob > 50.0:
        jsonarray = d
        return output.index(guess)
    else:
        return 1  # Return 'NOTHING' index

#%%
def initializers():
    imlist = modlistdir(path2)
    
    image1 = np.array(Image.open(path2 +'/' + imlist[0]))
    
    m,n = image1.shape[0:2]
    total_images = len(imlist)
    
    # create matrix to store all flattened images
    immatrix = np.array([np.array(Image.open(path2+ '/' + images).convert('L')).flatten()
                         for images in sorted(imlist)], dtype = 'f')
    
    print(immatrix.shape)
    
    input("Press any key")
    
    # Label the set of images per respective gesture type
    label=np.ones((total_images,),dtype = int)
    
    samples_per_class = int(total_images / nb_classes)
    print("samples_per_class - ",samples_per_class)
    s = 0
    r = samples_per_class
    for classIndex in range(nb_classes):
        label[s:r] = classIndex
        s = r
        r = s + samples_per_class
    
    data,Label = shuffle(immatrix,label, random_state=2)
    train_data = [data,Label]
     
    (X, y) = (train_data[0],train_data[1])
     
    # Split X and y into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
     
    X_train = X_train.reshape(X_train.shape[0], img_channels, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], img_channels, img_rows, img_cols)
     
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
     
    # normalize
    X_train /= 255
    X_test /= 255
     
    # convert class vectors to binary class matrices
    Y_train = to_categorical(y_train, nb_classes)
    Y_test = to_categorical(y_test, nb_classes)
    
    return X_train, X_test, Y_train, Y_test


def trainModel(model):
    # Split X and y into training and testing sets
    X_train, X_test, Y_train, Y_test = initializers()

    # Now start the training of the loaded model
    hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
                 verbose=1, validation_split=0.2)

    ans = input("Do you want to save the trained weights - y/n ?")
    if ans == 'y':
        filename = input("Enter file name - ")
        fname = path + str(filename) + ".weights.h5"
        model.save_weights(fname, overwrite=True)
    else:
        model.save_weights("0-newWeight.weights.h5", overwrite=True)
        
    visualizeHis(hist)
    visualize_training_history(hist)

#%%
def visualizeHis(hist):
    # visualizing losses and accuracy
    train_loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    train_acc = hist.history['accuracy']  # Changed from 'acc'
    val_acc = hist.history['val_accuracy']  # Changed from 'val_acc'
    
    xc = range(nb_epoch)

    plt.figure(1, figsize=(7,5))
    plt.plot(xc, train_loss)
    plt.plot(xc, val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train','val'])

    plt.figure(2, figsize=(7,5))
    plt.plot(xc, train_acc)
    plt.plot(xc, val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train','val'], loc=4)

    plt.savefig('./history/training_history.png')
    # plt.show()


def visualize_training_history(history) -> None:
    """
    Creates plots showing how the model improved during the training.

    Args:
        history: History object returned by model.fit()
    """
    # extract training metrics
    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    train_acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    epochs = range(1, len(train_loss) + 1)

    # create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # plot loss
    ax1.plot(epochs, train_loss, "-b", label="Training Loss")
    ax1.plot(epochs, val_loss, "r-", label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training vs Validation Loss")
    ax1.legend()
    ax1.grid(True)

    # Plot accuracy
    ax2.plot(epochs, train_acc, "b-", label="Training Accuracy")
    ax2.plot(epochs, val_acc, "r-", label="Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training vs Validation Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("./history/training_history_all.png")
    print("Training hisotry saved to: ./history/training_history_all.png")
    
    # We are in a non-interactive backend, so we don't call plt.show()
    # plt.show()

#%%
def visualizeLayers(model):
    imlist = modlistdir('./imgs')
    if len(imlist) == 0:
        print('Error: No sample image file found under \'./imgs\' folder.')
        return
    else:
        print('Found these sample image files - {}'.format(imlist))

    img = int(input("Which sample image file to load (enter the INDEX, starting from 0): "))
    layerIndex = int(input("Enter which layer to visualize. Enter -1 to visualize all layers: "))
    
    if img <= len(imlist):
        image = np.array(Image.open('./imgs/' + imlist[img]).convert('L')).flatten()
        
        ## Predict
        print('Guessed Gesture is {}'.format(output[guessGesture(model, image)]))
        
        # reshape it
        image = image.reshape(img_channels, img_rows, img_cols)
        
        # float32
        image = image.astype('float32')
        
        # normalize it
        image = image / 255
        
        # reshape for NN
        input_image = image.reshape(1, img_channels, img_rows, img_cols)
    else:
        print('Wrong file index entered !!')
        return
    
    if layerIndex >= 1:
        visualizeLayer(model, img, input_image, layerIndex)
    else:
        tlayers = len(model.layers[:])
        print("Total layers - {}".format(tlayers))
        for i in range(1, tlayers):
             visualizeLayer(model, img, input_image, i)

#%%
def visualizeLayer(model, img, input_image, layerIndex):
    layer = model.layers[layerIndex]
    
    # get_activations = K.function([model.layers[0].input], [layer.output])
    # activations = get_activations([input_image])[0]
    # output_image = activations

    # Create a new model that outputs the desired layer's activation
    activation_model = Model(inputs=model.inputs, outputs=layer.output)
    activations = activation_model.predict(input_image)
    output_image = activations

    ## If 4 dimensional then take the last dimension value as it would be no of filters
    if output_image.ndim == 4:
        output_image = np.moveaxis(output_image, 1, 3)
        
        print("Dumping filter data of layer{} - {}".format(layerIndex, layer.__class__.__name__))
        filters = output_image.shape[3]
        
        fig = plt.figure(figsize=(8,8))
        for i in range(min(filters, 36)):  # Limit to 36 filters max
            ax = fig.add_subplot(6, 6, i+1)
            ax.imshow(output_image[0,:,:,i], cmap='gray')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
        plt.tight_layout()
        
        savedfilename = "img_" + str(img) + "_layer" + str(layerIndex) + "_" + layer.__class__.__name__ + ".png"
        fig.savefig(savedfilename)
        print("Created file - {}".format(savedfilename))
    else:
        print("Can't dump data of this layer{} - {}".format(layerIndex, layer.__class__.__name__))
