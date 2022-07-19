'''
using VGG16: model that won 2014 ImageNet competition
-size of the full VGG16 network on disk is about 553 megabytes
Trained to classify images in 1000 categories
code below will: 
PREPROCESS
COMPILE
TRAIN
PREDICT
PLOT confusion matric
'''
import CNNImageProcessing_ex as cip
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from keras.optimizers import adam_v2
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
#useful function to plot confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
'''
NOTE: cip = CNNImageProcessing_ex.py (see file for cip.methods())
'''
#see cip file for more info
imgs, labels = next(cip.train_batches)
cip.plotImages(imgs)
print(labels)
#imprt the VGG16 model from Keras (save weights, parameters, etc)
#NOTE: requires internet connection
vgg16_model = tf.keras.applications.vgg16.VGG16()
#see summary of model
vgg16_model.summary()
#see type of model (Functional)
print(type(vgg16_model))
'''
FINE TUNING IMPORTED MODEL TO WORK FOR OUR NEEDS
convert functional type model to sequential
iterate over each layer in Functional model and add the layer to the sequential model
NOTE: last layer is not added (output will be different)
'''
model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)
#iterate over each layer in sequential and make them non-trainable
for layer in model.layers:
    layer.trainable = False
#add new output layer (2 nodes, softmax )
model.add(Dense(units=2, activation='softmax'))
model.summary()
#compile the model to use
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
#train the model for 5 epochs
#NOTE: to change what you see during training:https://stackoverflow.com/questions/39124676/show-progress-bar-for-each-epoch-during-batchwise-training-in-keras
model.fit(
          x=cip.train_batches,
          steps_per_epoch=len(cip.train_batches),
          validation_data=cip.valid_batches,
          validation_steps=len(cip.valid_batches),
          epochs=5,
          verbose=2
)
#get a batch of test samples and labels
#plot them to see what they look like
test_imgs, test_labels = next(cip.test_batches)
cip.plotImages(test_imgs)
print(test_labels)
#predict on the test data
predictions = model.predict(x=cip.test_batches, steps=len(cip.test_batches), verbose=0)
#plot the confusion matrix
cm = confusion_matrix(y_true=cip.test_batches.classes, y_pred=np.argmax(predictions, axis=-1))
cm_plot_labels = ['cat','dog']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')