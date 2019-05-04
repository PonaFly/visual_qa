import numpy as np
import tensorflow as tf
from PIL import Image
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.layers import Dropout, Flatten, Dense, Concatenate, Input, GRU, Embedding, LSTM
from keras.applications import ResNet50, VGG19
from keras.optimizers import Adam
from keras.models import  Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import json

def save_model(model,history,model_name):
    model_name = 'models/' + model_name    
    if  not os.path.exists('models'):
        os.system('mkdir models')        
    model.save_weights(model_name + '_model_weights.h5')
    model_json = model.to_json()
    with open(model_name + '_model.json', "w") as json_file:
        json_file.write(model_json)
    with open(model_name + '_hitory.json', 'w') as f:
        json.dump(history.history, f)

        
def clear_session():
    K.get_session().graph.get_collection('variables')
    K.clear_session()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    
def datagen(data_train,batch_size):
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

        
def old_model_construct(vocabulary_len,max_text_length,lr=1e-5,cnn='resnet'):  #vocabulary_len=12629, max_text_length=26
 
    img_shape = (224,224,3)
    img_input = Input(img_shape, name='image_input')
    if cnn == 'resnet':
        cnn_base = ResNet50(weights='imagenet', input_tensor=img_input, include_top=False)     # 175 layers
        cnn_flatten = Flatten(name='cnn_flatten')(cnn_base.get_layer('activation_49').output)
    else:
        cnn_base = VGG19(weights='imagenet', input_tensor=img_input, include_top=False)
        cnn_flatten = Flatten(name='cnn_flatten')(cnn_base.get_layer('block5_pool').output)
    for l in cnn_base.layers:
            l.trainable = True
    
    EMBEDDING_DIM = 100
    text_input = Input((max_text_length,))
    embed = Embedding(input_dim=vocabulary_len, output_dim=EMBEDDING_DIM, input_length=max_text_length)(text_input)
    drop_1 = Dropout(0.5)(embed)
    lstm_1 = LSTM(512, input_shape=(max_text_length, EMBEDDING_DIM),return_sequences=True)(drop_1)
    drop_2 = Dropout(0.5)(lstm_1)
    lstm_2 = LSTM(512,activation='sigmoid',return_sequences=True)(drop_2)
    drop_3 = Dropout(0.5)(lstm_2)
    dense_1 = Dense(1024,activation='tanh')(drop_3)
    lstm_flatten = Flatten(name='gru_flatten')(dense_1)
    data_concatenate = Concatenate(name='all_data_concat')([cnn_flatten,lstm_flatten])
    dense_2 = Dense(1024,activation='tanh')(data_concatenate)
    drop_4 = Dropout(0.5)(dense_2)
    dense_2 = Dense(1,activation='softmax')(drop_4)
    
    model = Model([img_input,text_input], dense_2)
    model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy',metrics=[f1,'accuracy']) 
    print('model construced!')
    
    return model

def new_model_construct(vocabulary_len,max_text_length,lr=1e-5,cnn='resnet'):  #kostil
 
    img_shape = (224,224,3)
    img_input = Input(img_shape, name='image_input')
    if cnn == 'resnet':
        cnn_base = ResNet50(weights='imagenet', input_tensor=img_input, include_top=False)     # 175 layers
        cnn_flatten = Flatten(name='cnn_flatten')(cnn_base.get_layer('activation_49').output)
    else:
        cnn_base = VGG19(weights='imagenet', input_tensor=img_input, include_top=False)
        cnn_flatten = Flatten(name='cnn_flatten')(cnn_base.get_layer('block5_pool').output)
    for l in cnn_base.layers:
            l.trainable = True
    
    EMBEDDING_DIM = 500
    text_input = Input((max_text_length,))
    embed = Embedding(input_dim=vocabulary_len, output_dim=EMBEDDING_DIM, input_length=max_text_length)(text_input)
    drop_1 = Dropout(0.5)(embed)
    lstm_1 = LSTM(1024, input_shape=(max_text_length, EMBEDDING_DIM),return_sequences=True)(drop_1)
    drop_2 = Dropout(0.5)(lstm_1)
    lstm_2 = LSTM(1024,return_sequences=True)(drop_2)
    drop_3 = Dropout(0.5)(lstm_2)
    dense_1 = Dense(1024,activation='tanh')(drop_3)
    lstm_flatten = Flatten(name='gru_flatten')(dense_1)
    data_concatenate = Concatenate(name='all_data_concat')([cnn_flatten,lstm_flatten])
    dense_2 = Dense(1024,activation='tanh')(data_concatenate)
    drop_4 = Dropout(0.5)(dense_2)
    dense_2 = Dense(1,activation='softmax')(drop_4)
    
    model = Model([img_input,text_input], dense_2)
    model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy',metrics=[f1,'accuracy']) 
    print('model construced!')
    
    return model


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def get_history_params(history,es_patience,*params):
    params_values = {}
    for i in range(es_patience,-1,-1):
        try:
            for param in params:
                params_values[param] = str(round(history.history[param][-(i+1)],3))
            return params_values
        except:
            continue

def fit(model,data_train,epochs,batch_size,steps,es_patience):
    #checkpoint = ModelCheckpoint('retrained_fourth-{epoch:02d}-{loss:.4f}.h5', monitor='loss', verbose=1, 
    #                             save_best_only=True, mode='min',save_weights_only=True)
    data_gen = datagen(data_train, batch_size)
    history = model.fit_generator(data_gen,steps_per_epoch=steps, 
        epochs=epochs, 
        verbose=1,workers=3,use_multiprocessing=True,
        callbacks=[EarlyStopping(monitor='loss',patience=es_patience,restore_best_weights=True)],
    )
    return history

clear_session()



