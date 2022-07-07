'''
simple artificial neural network using a Sequential model from Keras API
NO GPU usage is used or set up in this code
GPU SET UP: https://deeplizard.com/learn/video/IubEtS2JAiY
source: https://deeplizard.com/learn/video/Boo6SmgmHuM

'''
import dataProcessing_ex as dpEx
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.optimizers import adam_v2
from keras.metrics import categorical_crossentropy
'''

setting up the layers of the NN 
NOTE: First layer is the input data itself
model below is 4 layers

units: the number of nodes in the layer

output layer: last layer in the model should have as many nodes as classifications

'''
model = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')
])
#print the model details for each layer (Layer type/ Output Shape/ parameters)
print(model.summary())
#model needs to be compiled to use
#Adam optimization is a stochastic gradient descent (SGD) method
'''
NOTE: we have only two classes, we could instead configure our output layer to have only one output
      and use binary_crossentropy as our loss 
      Both options work equally well and achieve the exact same result.
'''
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#train model use fit
#NOTE: all NN's do is create a curve that 'fits' the training data well
       #that 'fit curve' is then used to make predictions/classifications
#x = sample data
#y = sample data labels
#validation split = percent of samples set aside
#batch_size = number of samples passed at a time
#epochs = number of passes of all batches through the network
#verbose (0-2) amount of output information we see 
model.fit(
    x=dpEx.scaled_train_samples
    , y=dpEx.train_labels
    , validation_split = 0.1
    , batch_size=10
    , epochs=30
    , verbose=2)
#MAKING PREDICTIONS
'''
test set was made in dataProcessing_ex.py file
print predictions so see them
'''
predictions = model.predict(
      x=dpEx.scaled_test_samples
    , batch_size=10
    , verbose=0
)  
rounded_predictions = np.argmax(predictions, axis=-1)

if __name__ == "__main__":
    print("predictions raw")
    for i in predictions:
        print(i)
    print("most probable prediction")
    for i in rounded_predictions:
        print(i)