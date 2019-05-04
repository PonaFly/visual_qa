#loss 8.0385 f1 0.6591 acc 0.4958 default params in Model
import h5py
import json
import Model
import argparse

def main(params):
    model_name = params.name
    EPOCHS = params.epochs
    BATCH_SIZE = params.bs
    es_patience = params.es

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
    if params.model_type == 1:
        model = Model.old_model_construct(len(vocabulary),data_train['ques_train'].shape[1],params.lr,params.cnn)
    else:
        model = Model.new_model_construct(len(vocabulary),data_train['ques_train'].shape[1],params.lr,params.cnn)
    if params.steps == -1:
        steps = len(data_train['answers']) // BATCH_SIZE
    else:
        steps = params.steps
    print('number of steps per epoch: ',steps)

    print('fitting')
    history = Model.fit(model,data_train,EPOCHS, BATCH_SIZE, steps,es_patience)

    print('model saving')
    history_params = Model.get_history_params(history,es_patience,'loss','f1')
    model_name = model_name + '_loss:' + history_params['loss'] + ';f1:' + history_params['f1']
    Model.save_model(model,history,model_name)

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(add_help=True)
    
    parser.add_argument('--name', required=True, help='model name')
    parser.add_argument('--cnn', default='resnet', help='vgg19 or resnet50 as pretrained cnn')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--bs', default=50, type=int)
    parser.add_argument('--es', default=2, type=int, help='EarlyStopping callback patience')
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--steps', default=-1, type=int, help='only for test')
    parser.add_argument('--model_type', default=1, type=int, help='check model.py')
    args = parser.parse_args()
    
    print('parsed input parameters:')
    print(json.dumps(vars(args), indent = 5))
    main(args)    

