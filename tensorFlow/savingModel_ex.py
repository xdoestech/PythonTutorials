'''
ways to save and load model for future use 
Source: https://deeplizard.com/learn/video/7n1SpeudvAE
'''
import NNKerasAPI_ex as nnk

'''
Save everything about model
the architecture, 
the weights, 
the optimizer, 
the state of the optimizer, 
the learning rate, 
the loss, etc.
'''
nnk.model.save('FILE PATH TO SAVE FILE.h5')
from keras.models import load_model
new_model = load_model('SAME PATH AS ABOVE.h5')
#verify models have same architecture and weights by viewing both summaries
new_model.summary()
nnk.model.summary()

'''
Save only the architecture of the model
NO WEIGHTS,
NO CONFIGS,
NO OPTIMIZER,
NO LOSS,
JUST LAYER STRUCTURE
'''
#save architecture as json
json_string = nnk.model.to_json()
#load model from json
from keras.models import model_from_json
model_architecture = model_from_json(json_string)
#save architecture as yaml
    #this is no longer supported
'''
Save only the weights of model
New model eeds to have the same architecture as old model
'''
nnk.model.save_weights('FILE PATH TO SAVE.h5')
#load saved weights into new model with same architecture
model2 = nnk.Sequential([
    nnk.Dense(units=16, input_shape=(1,), activation='relu'),
    nnk.Dense(units=32, activation='relu'),
    nnk.Dense(units=2, activation='softmax')
])

model2.load_weights('models/my_model_weights.h5')

