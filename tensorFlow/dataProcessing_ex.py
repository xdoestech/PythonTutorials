'''
example creating, processing, printing data
source: https://deeplizard.com/learn/video/UkzhouEk6uY
'''
import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
#CREATING TRAINING DATA SAMPLE SET
train_labels = []
train_samples = []
'''
let's suppose that an experimental drug was tested on individuals ranging from age 13 to 100 in a clinical trial.
The trial had 2100 participants.
Half of the participants were under 65 years old, and the other half was 65 years of age or older.

The trial showed that around 95% of patients 65 or older experienced side effects from the drug, 
and around 95% of patients under 65 experienced no side effects, 
generally showing that elderly individuals were more likely to experience side effects.
'''
#random_younger/random_older: int representing age
#age in train_samples
# side effects(1) or no side effects(0) in train_labesl
for i in range(50):
    # The ~5% of younger individuals who did experience side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(1)

    # The ~5% of older individuals who did not experience side effects
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(0)

for i in range(1000):
    # The ~95% of younger individuals who did not experience side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(0)

    # The ~95% of older individuals who did experience side effects
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(1)

#convert lists to numpy arrays
#shuffle arrays to ensure randomish order
train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
train_labels, train_samples = shuffle(train_labels, train_samples)
#scale data into range between 0 - 1
#this is a required format for the model we will use
scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))

#CREATING TESTING DATA SAMPLE SET
test_labels =  []
test_samples = []

for i in range(10):
    # The 5% of younger individuals who did experience side effects
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(1)

    # The 5% of older individuals who did not experience side effects
    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(0)

for i in range(200):
    # The 95% of younger individuals who did not experience side effects
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(0)

    # The 95% of older individuals who did experience side effects
    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(1)

test_labels = np.array(test_labels)
test_samples = np.array(test_samples)
test_labels, test_samples = shuffle(test_labels, test_samples)

scaled_test_samples = scaler.fit_transform(test_samples.reshape(-1,1))

#DISPLAY THE CREATED SAMPLES
if __name__ == "__main__":
    print("==========TRAIN SAMPLES===========")
    for i in train_samples:
        print(i)
    for i in train_labels:
        print(i)
    for i in scaled_train_samples:
        print(i)
    print("==========TEST SAMPLES===========")
    for i in test_samples:
        print(i)
    for i in test_labels:
        print(i)
    for i in scaled_test_samples:
        print(i)