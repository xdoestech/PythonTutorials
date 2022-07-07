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

#this CNN will use a sequential model
#Two Convolutional Layers
#one Dense Layer (output)
#input layer is the data itself(not defined here)
model = Sequential([
    #zero padding (same)
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding = 'same', input_shape=(224,224,3)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(units=2, activation='softmax')
])
'''
See File NNKerasAPI_ex.py for more info on below
steps_per_epoch: # of batches passed through model in each epoch
NOTE: cip= CNNImageProcessing_ex.py (see file for methods)

'''
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
#set up model to train
model.fit(x=cip.train_batches,
    steps_per_epoch=len(cip.train_batches),
    validation_data=cip.valid_batches,
    validation_steps=len(cip.valid_batches),
    epochs=10,
    verbose=2
)
'''
Making predictions with the model
'''
test_imgs, test_labels = next(cip.test_batches)
cip.plotImages(test_imgs)
print(test_labels)

predictions = model.predict(x=cip.test_batches, steps=len(cip.test_batches), verbose=0)

np.round(predictions)
cm = confusion_matrix(y_true=cip.test_batches.classes, y_pred=np.argmax(predictions, axis=-1))

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
#see how classes are index
cip.test_batches.class_indices
#plot the data
cm_plot_labels = ['cat','dog']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
