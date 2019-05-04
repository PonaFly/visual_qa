
python3 train.py --name resnet_default
python3 train.py --name vgg_default

python3 train.py --name resnet_default_low_lr --lr 1e-6
python3 train.py --name vgg_default_low_lr  --cnn vgg --lr 1e-6

python3 train.py --name resnet_default_large_lr  --lr 1e-4
python3 train.py --name vgg_default_large_lr  --cnn vgg --lr 1e-4

python3 train.py --name resnet_new --model_type 2 
python3 train.py --name vgg_new --model_type 2 --cnn vgg

python3 train.py --name resnet_new_low_lr --model_type 2 --lr 1e-6
python3 train.py --name vgg_new_low_lr --model_type 2 --cnn vgg --lr 1e-6

python3 train.py --name resnet_new_large_lr --model_type 2 --lr 1e-4
python3 train.py --name vgg_new_large_lr --model_type 2 --cnn vgg --lr 1e-4