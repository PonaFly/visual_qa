#loss 8.0385 f1 0.6591 acc 0.4958

import json
import numpy as np
import h5py
import tensorflow as tf
from PIL import Image
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.layers import Dropout, Flatten, Dense, Concatenate, Input, GRU, Embedding, LSTM
from keras.applications import ResNet50
from keras.optimizers import Adam
from keras.models import  Model
from keras.callbacks import ModelCheckpoint, EarlyStopping


def save_model(model,model_name):
    model.save_weights(model_name + '_model_weights.h5')
    model_json = model.to_json()
    with open(model_name + '_model.json', "w") as json_file:
        json_file.write(model_json)

def clear_session():
    K.get_session().graph.get_collection('variables')
    K.clear_session()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))
    


def datagen(batch_size):
    data_copy = {'ques_train':np.array([])}
    while True:
        if data_copy['ques_train'].shape[0] == 0:
            data_copy = data_train.copy()
        
        img_links = data_copy['img_link'][:batch_size]
        imgs = []
        for link in img_links:
            img = Image.open('data/'+link)
            rsize = img.resize((224,224))    
            rsizeArr = np.asarray(rsize)     
            
            if rsizeArr.shape == (224,224):
                rsizeArr = np.dstack([rsizeArr]*3)
                
            imgs.append(rsizeArr)
        ques = data_copy['ques_train'][:batch_size]
        answers = data_copy['answers'][:batch_size]
        imgs = np.array(imgs) / 255
        
        data_copy['ques_train'] = data_copy['ques_train'][batch_size:]
        data_copy['answers'] = data_copy['answers'][batch_size:]
        data_copy['img_link'] = data_copy['img_link'][batch_size:]
        yield [np.array(imgs), ques], answers


def model_construct():
 
    img_shape = (224,224,3)
    img_input = Input(img_shape, name='image_input')
    res_base = ResNet50(weights='imagenet', input_tensor=img_input, include_top=False)     # 175 layers
    
    for l in res_base.layers:
            l.trainable = True

    resn_flatten = Flatten(name='resn_flatten')(res_base.get_layer('activation_49').output)
    
    EMBEDDING_DIM = 100
    VOCABULARY_SIZE = len(vocabulary) # 12629
    MAX_TEXT_LENGTH = data_train['ques_train'].shape[1] # 26
    input_length=MAX_TEXT_LENGTH
    text_input = Input((input_length,))
    embed = Embedding(input_dim=VOCABULARY_SIZE, output_dim=EMBEDDING_DIM, input_length=input_length)(text_input)
    drop_1 = Dropout(0.5)(embed)
    #gru_1 = GRU(1024, input_shape=(MAX_TEXT_LENGTH, EMBEDDING_DIM),return_sequences=True)(drop_1)
    gru_1 = LSTM(512, input_shape=(MAX_TEXT_LENGTH, EMBEDDING_DIM),return_sequences=True)(drop_1)
    drop_2 = Dropout(0.5)(gru_1)
    #gru_2 = GRU(1024,activation='sigmoid',return_sequences=True)(drop_2)
    gru_2 = LSTM(512,activation='sigmoid',return_sequences=True)(drop_2)
    drop_3 = Dropout(0.5)(gru_2)
    dense_1 = Dense(1024,activation='tanh')(drop_3)
    gru_flatten = Flatten(name='gru_flatten')(dense_1)
    data_concatenate = Concatenate(name='all_data_concat')([resn_flatten,gru_flatten])
    dense_2 = Dense(1024,activation='tanh')(data_concatenate)
    drop_4 = Dropout(0.5)(dense_2)
    dense_2 = Dense(1,activation='softmax')(drop_4)
    
    model = Model([img_input,text_input], dense_2)
    model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy',metrics=[f1,'accuracy']) 
    print('model construced!')
    return model



def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# In[6]:


data_train = {}
with h5py.File('data/data_prepro_train.h5', 'r') as f:
    for key in f.keys():
        data_train[key] = f[key][:]


# In[7]:


with open('data/data_prepro_train.json') as json_file:  
    data_train_info = json.load(json_file)


# In[8]:


data_train['img_link'] = []
for pos in data_train['img_pos_train']:
    data_train['img_link'].append(data_train_info['unique_img_train'][pos])


# In[9]:


del data_train['img_pos_train']
del data_train['ques_length_train']
del data_train['question_id_train']


# In[10]:


with open('data/vocab.json') as json_file:  
    vocabulary = json.load(json_file)


# In[11]:




# In[12]:


def fit(epochs,batch_size,steps):
    #checkpoint = ModelCheckpoint('retrained_fourth-{epoch:02d}-{loss:.4f}.h5', monitor='loss', verbose=1, 
    #                             save_best_only=True, mode='min',save_weights_only=True)
    data_gen = datagen(batch_size)
    model.fit_generator(data_gen,steps_per_epoch=steps, 
        epochs=epochs, 
        verbose=1,workers=3,use_multiprocessing=True,
        callbacks=[EarlyStopping(monitor='loss',patience=2,restore_best_weights=True)],
    )


# In[13]:


clear_session()


# In[14]:


# TensorFlow wizardry
config = tf.ConfigProto()
 
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
 
# Only allow a total of half the GPU memory to be allocated
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
 
# Create a session with the above options specified.
K.tensorflow_backend.set_session(tf.Session(config=config))


# In[15]:


model = model_construct()


# In[17]:


epochs = 100
batch_size = 50
steps = len(data_train['answers']) // batch_size
print(steps)


# In[18]:


history = fit(epochs,batch_size,steps)



MODEL_NAME = 'new_long_train' #lr=4 100 steps


# In[20]:


with open(MODEL_NAME + '_hitory.json', 'w') as f:
        json.dump(history, f)

save_model(model,MODEL_NAME)


# In[ ]:




