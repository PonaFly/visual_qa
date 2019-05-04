#loss 8.0385 f1 0.6591 acc 0.4958
import h5py
import json
import Model

MODEL_NAME = 'new_long_train'
EPOCHS = 100
BATCH_SIZE = 50

data_link = 'data/data_prepro_train.'
data_train = {}
data_train['img_link'] = []

print('data loading...')

with h5py.File(data_link+'h5', 'r') as f:
    for key in f.keys():
        data_train[key] = f[key][:]
        
with open(data_link+'json') as json_file:  
    data_train_info = json.load(json_file)

with open('data/vocab.json') as json_file:  
    vocabulary = json.load(json_file)
    
for pos in data_train['img_pos_train']:
    data_train['img_link'].append(data_train_info['unique_img_train'][pos])

del data_train['img_pos_train']
del data_train['ques_length_train']
del data_train['question_id_train']

print('constructing model...')
model = Model.model_construct(len(vocabulary),data_train['ques_train'].shape[1])

steps = len(data_train['answers']) // BATCH_SIZE
print('number of steps per epoch: ',steps)

print('fitting')
history = Model.fit(model,data_train,EPOCHS, BATCH_SIZE, steps)

print('model saving')
Model.save_model(model,MODEL_NAME)

with open(MODEL_NAME + '_hitory.json', 'w') as f:
        json.dump(history, f)
