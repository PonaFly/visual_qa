## data manipulation:
1. Run `python3 vqa_preprocessing.py --download --binary` to download data and filter out questions with yes/no answer.
2.  Run `python3 dataset_prepro.py --num_ans 2` if you work with binary classification, else change `--num_ans` arg

## train:
1. Run `python3 train.py` with changing params in script, if you want

## evaluate:
In progress

## evaluating telegram bot:
In progress


#### some code for data preprocessing is taken from https://github.com/GT-Vision-Lab/VQA_LSTM_CNN