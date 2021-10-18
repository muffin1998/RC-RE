from data_utils.data_process import process_semeval
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased',
                        help='pre-trained bert model to load')
    parser.add_argument('--data_dir', type=str,
                        help='dataset path')
    parser.add_argument('--task', type=str,
                        help='task name')
    args = parser.parse_args()
    if args.task == 'semeval':
        process_semeval(dataset_path=args.data_dir, bert_model=args.bert_model)

