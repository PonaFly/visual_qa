## data manipulation:
1. Run `python3 vqa_preprocessing.py --download --binary` to download data and filter out questions with 'yes/no' answer.
2.  Run `python3 dataset_prepro.py --num_ans 2` if you work with binary classification, else change `--num_ans` arg

`python3 vqa_preprocessing.py -h` or `python3 dataset_prepro.py -h` to check which params you can change

## train:
Run `python3 train.py --name *model name*`

Also you can change some params: 
`python3 train.py -h`

## evaluate:
In progress

## evaluating telegram bot:
In progress


##### some code for data preprocessing is taken from https://github.com/GT-Vision-Lab/VQA_LSTM_CNN and by pull req.  https://github.com/varunagrawal
