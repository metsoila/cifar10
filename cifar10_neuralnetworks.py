import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import random
from random import randint
import statistics
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils


def class_acc(pred,gt):
    
    samples = pred.shape[0]
    match = 0
    for i in range(0, samples):
        if (pred[i] == gt[i]):
            match += 1
    print(f'acc. {match/samples *100 :.2f} %')
    return 0



def cifar_10_reshape(batch_arg):
    
    output = batch_arg.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
    return output



def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin')
    return dict

def load_batch_file(batch_filename):
    filepath = os.path.join('../input/cifar-10-batches-py/', batch_filename)
    unpickled = unpickle(filepath)
    return unpickled


def load_data():
    
    folder = 'cifar-10-batches-py/'
    trainImages = []
    trainLabels = []
    testData = []
    
    for i in range(1,6):
        batch = unpickle((folder + 'data_batch_{}'.format(i)))
        data = cifar_10_reshape(batch['data'])
        labels = batch['labels']
        trainImages.append(data)
        trainLabels.append(labels)
        
    test_batch = unpickle(folder + 'test_batch')
    testImages = cifar_10_reshape(test_batch['data'])
    testLabels = test_batch['labels']
    
    testData.append([testImages, testLabels])
    trainData = [trainImages,trainLabels]
    
    return trainData, testData





def cifar10_color(trainData):
    
    x = trainData[0][0]
    
    means_ = np.mean(x, axis=(0, 1))
    vars_ = np.var(x, axis = (0,1))

    Xp = [means_, vars_]
    return Xp

def plot_history(history, title):
    plt.figure(figsize=(10,3))
    # Plot training & validation accuracy values
    plt.subplot(121)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

def main():
        
    trainData, testData = load_data()

    #labels
    labeldict = unpickle('cifar-10-batches-py/batches.meta')
    label_names = labeldict["label_names"]
    Xp = cifar10_color(trainData)
    
    print(Xp)
    
    num_classes = 10
    
    train_batch_1 = load_batch_file('data_batch_1')
    train_batch_2 = load_batch_file('data_batch_2')
    train_batch_3 = load_batch_file('data_batch_3')
    train_batch_4 = load_batch_file('data_batch_4')
    train_batch_5 = load_batch_file('data_batch_5')
    test_batch = load_batch_file('test_batch')
    
    train_x = np.concatenate([train_batch_1['data'], train_batch_2['data'], train_batch_3['data'], train_batch_4['data'], train_batch_5['data']])
    train_x = train_x.astype('float32') # this is necessary for the division below
    train_x /= 255
    train_y = np.concatenate([np_utils.to_categorical(labels, num_classes) for labels in [train_batch_1['labels'], train_batch_2['labels'], train_batch_3['labels'], train_batch_4['labels'], train_batch_5['labels']]])
    
    test_x = test_batch['data'].astype('float32') / 255
    test_y = np_utils.to_categorical(test_batch['labels'], num_classes)
    
    
    img_rows = img_cols = 32
    channels = 3
    
    trainImages = trainData[0][0]
    trainLabels = trainData[1][0]
    
    simple_model = Sequential()
    simple_model.add(Dense(10_000, input_shape=(img_rows*img_cols*channels,), activation='relu'))
    simple_model.add(Dense(10, activation='softmax'))
    
    simple_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    simple_model_history = simple_model.fit(train_x, train_y, batch_size=100, epochs=8, validation_data=(test_x, test_y))
    

    
    
    
    for i in range(trainImages.shape[0]):
        # Show some images randomly
        if random() > 0.999:
            plt.figure(1)
            plt.clf()
            plt.imshow(trainImages[i])
            plt.title(f"Image {i} label={label_names[trainLabels[i]]} (num {trainLabels[i]})")
            plt.pause(0.01)
            
    return 0

if __name__ == "__main__":
    main()

