# RC-RE

## 1. Data Preprocess

Take semeval dataset as example:

```bash
python preprocess.py --task semeval --data_dir dataset/semeval --bert_model bert-base-uncased
```

Make sure the dataset files are named with {train|dev|test}.txt and put under dataset/{data_dir}/raw/, and the labels.txt should be put under dataset/{data_dir}/.

## 2. Train and Evaluate
```bash
python main.py --do_train --do_eval
```
Avaialbe arguments:
```
usage: main.py [-h] [--bert_model BERT_MODEL] [--task TASK]
               [--data_dir DATA_DIR] [--pos_weight POS_WEIGHT]
               [--hidden_state_unit HIDDEN_STATE_UNIT] [--epochs EPOCHS]
               [--seq_len SEQ_LEN] [--weight_decay_rate WEIGHT_DECAY_RATE]
               [--learning_rate LEARNING_RATE] [--warmup_steps WARMUP_STEPS]
               [--batch_size BATCH_SIZE] [--num_class NUM_CLASS]
               [--num_relation NUM_RELATION] [--seed SEED]
               [--bin_weight BIN_WEIGHT] [--do_train] [--do_eval]
               [--overwrite]

optional arguments:
  -h, --help            show this help message and exit
  --bert_model BERT_MODEL
                        pre-trained bert model to load
  --task TASK           task name
  --data_dir DATA_DIR   dataset path
  --pos_weight POS_WEIGHT
                        positive weight in binary classification
  --hidden_state_unit HIDDEN_STATE_UNIT
                        the size of hidden state unit
  --epochs EPOCHS       max epochs
  --seq_len SEQ_LEN     max sequence length
  --weight_decay_rate WEIGHT_DECAY_RATE
                        the rate of weight decay
  --learning_rate LEARNING_RATE
                        initial learning rate
  --warmup_steps WARMUP_STEPS
                        the number of warmup step
  --batch_size BATCH_SIZE
                        the size of mini-batch
  --num_class NUM_CLASS
                        number of class
  --num_relation NUM_RELATION
                        number of relation
  --seed SEED           random seed
  --bin_weight BIN_WEIGHT
                        weight of binary task
  --do_train            do train
  --do_eval             do eval
  --overwrite           overwrite existing checkpoint
```

## 3. Evaluate Trained Model
The trained model can be downloaded in:

https://mayear-my.sharepoint.com/:u:/g/personal/muffin_mayear_onmicrosoft_com/EboRcgexDO9KuyN-XSH84ZEB27QqYKu9r8avNFVB9217yQ?e=pcInO7

or

https://drive.google.com/file/d/1fyb-OrRN-fErVg37WYRbreSBvor_4mwf/view?usp=sharing

1. Put tensorflow checkpoint into saved_model with file name {task}
2. Run
```bash
python main.py --do_eval --task {task_name} --data_dir {data_dir} \
               --num_class {num_class} --num_relation {num_relation} \
               --seq_len {seq_len} --hidden_state_unit {hidden_state_unit}
```
