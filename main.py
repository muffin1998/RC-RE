import os
import tensorflow as tf
import random
import math
import numpy as np
import argparse
from typing import Optional, List, Union
from model.rc_bert import RCBert
from data_utils.dataset import load_dataset
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from transformers import AdamWeightDecay
from tqdm.std import tqdm


def accumulated_gradients(gradients: Optional[List[tf.Tensor]],
                          step_gradients: List[Union[tf.Tensor, tf.IndexedSlices]],
                          num_grad_accumulates: int) -> tf.Tensor:
    if gradients is None:
        gradients = [flat_gradients(g) / num_grad_accumulates for g in step_gradients]
    else:
        for i, g in enumerate(step_gradients):
            gradients[i] += flat_gradients(g) / num_grad_accumulates

    return gradients


# This is needed for tf.gather like operations.
def flat_gradients(grads_or_idx_slices: tf.Tensor) -> tf.Tensor:
    """Convert gradients if it's tf.IndexedSlices.
    When computing gradients for operation concerning `tf.gather`, the type of gradients
    """
    if type(grads_or_idx_slices) == tf.IndexedSlices:
        return tf.scatter_nd(
            tf.expand_dims(grads_or_idx_slices.indices, 1),
            grads_or_idx_slices.values,
            grads_or_idx_slices.dense_shape
        )
    return grads_or_idx_slices


def train(**kwargs):
    print(kwargs)
    bert_model = kwargs.pop('bert_model')
    batch_size = kwargs.pop('batch_size')
    learning_rate = kwargs.pop('learning_rate')
    hidden_state_unit = kwargs.pop('hidden_state_unit')
    weight_decay_rate = kwargs.pop('weight_decay_rate')
    max_seq_len = kwargs.pop('max_seq_len')
    max_epoch = kwargs.pop('max_epoch')
    seed = kwargs.pop('seed')
    pos_weight = kwargs.pop('pos_weight')
    data_dir = kwargs.pop('data_dir')
    num_class = kwargs.pop('num_class')
    num_relation = kwargs.pop('num_relation')
    bin_weight = kwargs.pop('bin_weight')
    warmup_steps = kwargs.pop('warmup_steps')

    def set_seed(s, delta):
        if s is not None:
            s += delta
        np.random.seed(s)
        random.seed(s)
        tf.random.set_seed(s)

    set_seed(seed, 0)
    (train_data, test_data, dataset_size) = load_dataset(data_dir, max_seq_len, num_class)

    model = RCBert(bert_model=bert_model,
                   hidden_state_unit=hidden_state_unit,
                   pos_weight=pos_weight,
                   num_relation=num_relation,
                   num_class=num_class)
    ce_fn = SparseCategoricalCrossentropy()
    train_steps = max_epoch * (math.ceil(dataset_size / batch_size))
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=train_steps - warmup_steps,
        end_learning_rate=learning_rate * 0.,
    )
    step_per_epoch = math.ceil(dataset_size / batch_size)
    validation_per_step = step_per_epoch // 2
    print(f"Validation per {validation_per_step} steps")
    optimizer = None
    for epoch in range(max_epoch):
        set_seed(seed, epoch + 10)
        batched_data = train_data.shuffle(dataset_size).batch(batch_size).prefetch(4)
        with tqdm(enumerate(batched_data), total=step_per_epoch) as t:
            t.set_description(f'Epoch: {epoch + 1}')
            for (step, batch) in t:
                global_step = epoch * step_per_epoch + step
                (inputs, labels) = batch
                with tf.GradientTape() as tape:
                    predictions, b_loss = model(inputs, training=True)
                    ce_loss = ce_fn(labels, predictions)
                    loss = tf.add_n([ce_loss, bin_weight * b_loss])
                trainable_vars = [v for v in model.trainable_variables if 'pooler' not in v.name]
                gradients = tape.gradient(loss, trainable_vars)

                if optimizer is None:
                    excluded_weights = [v.name for v in trainable_vars
                                        if 'bias' in v.name.lower() or
                                        'layernorm' in v.name.lower() or
                                        'layer_norm' in v.name.lower()]
                    include_weights = [v.name for v in trainable_vars if v.name not in excluded_weights]
                    optimizer = AdamWeightDecay(learning_rate=lr_schedule, epsilon=1e-8,
                                                beta_1=0.9, beta_2=0.999,
                                                weight_decay_rate=weight_decay_rate,
                                                include_in_weight_decay=include_weights,
                                                exclude_from_weight_decay=excluded_weights)

                optimizer.apply_gradients(zip(gradients, trainable_vars))

                accuracy = SparseCategoricalAccuracy()(labels, predictions)
                metrics = {
                    'Loss': loss.numpy(),
                    'Accuracy': accuracy.numpy()
                }
                t.set_postfix(metrics)

    return model


def evaluate(**kwargs):
    bert_model = kwargs.pop('bert_model')
    hidden_state_unit = kwargs.pop('hidden_state_unit')
    max_seq_len = kwargs.pop('max_seq_len')
    data_dir = kwargs.pop('data_dir')
    task = kwargs.pop('task')
    num_class = kwargs.pop('num_class')
    num_relation = kwargs.pop('num_relation')

    (_, test_data, _) = load_dataset(data_dir, max_seq_len, num_class)
    test_data = test_data.batch(32)

    model = RCBert(bert_model=bert_model,
                   hidden_state_unit=hidden_state_unit,
                   num_relation=num_relation,
                   num_class=num_class)
    model.load_weights(f'saved_model/{task}')
    y_pred = []
    y_true = []
    for batch in test_data:
        (inputs, labels) = batch
        predictions, bce_loss = model(inputs, training=False)
        y_pred.append(tf.argmax(predictions, axis=-1))
        y_true.append(labels)

    y_true = tf.concat(y_true, axis=0)
    y_pred = tf.concat(y_pred, axis=0)
    if task == 'semeval':
        _, full = get_semeval_result(data_dir, y_true, y_pred)

    print(full)


def get_semeval_result(data_dir, labels, predictions):
    import os
    import subprocess

    output_path = 'tmp'
    answer_path = os.path.join(output_path, 'answer.txt')
    prediction_path = os.path.join(output_path, 'prediction.txt')
    result_path = os.path.join(output_path, 'result.txt')
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    answer_file = open(answer_path, 'w')
    prediction_file = open(prediction_path, 'w')
    scorer_path = 'scripts/semeval2010_task8_scorer-v1.2.pl'

    with open(f'{data_dir}/labels.txt') as fp:
        label2relation = [relation.strip() for relation in fp]

    for (index, (label, prediction)) in enumerate(zip(labels, predictions)):
        answer_file.write(f'{index + 1}\t{label2relation[int(label.numpy())]}\n')
        prediction_file.write(f'{index + 1}\t{label2relation[int(prediction.numpy())]}\n')
    answer_file.close()
    prediction_file.close()

    while True:
        try:
            process = subprocess.Popen(["perl", scorer_path, prediction_path, answer_path],
                                       stdout=subprocess.PIPE)
            result_lines = [line.decode() for line in iter(process.stdout.readline, b'')]
            with open(result_path, 'w') as f:
                for (index, line) in enumerate(result_lines):
                    f.write(line)
                    if index == len(result_lines) - 1:
                        idx = line.find('%')
                        f1_score = float(line[idx - 5:idx])
                        full = result_lines[index - 4]
        except Exception as e:
            print(e)
        else:
            break

    return f1_score, full


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased',
                        help='pre-trained bert model to load')
    parser.add_argument('--task', type=str, default='semeval',
                        help='task name')
    parser.add_argument('--data_dir', type=str, default='dataset/semeval',
                        help='dataset path')
    parser.add_argument('--pos_weight', type=float, default=0.4,
                        help='positive weight in binary classification')
    parser.add_argument('--hidden_state_unit', type=int, default=768,
                        help='the size of hidden state unit')
    parser.add_argument('--epochs', type=int, default=7,
                        help='max epochs')
    parser.add_argument('--seq_len', type=int, default=128,
                        help='max sequence length')
    parser.add_argument('--weight_decay_rate', type=float, default=0.0,
                        help='the rate of weight decay')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='initial learning rate')
    parser.add_argument('--warmup_steps', type=int, default=0,
                        help='the number of warmup step')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='the size of mini-batch')
    parser.add_argument('--num_class', type=int, default=19,
                        help='number of class')
    parser.add_argument('--num_relation', type=int, default=9,
                        help='number of relation')
    parser.add_argument('--seed', type=int,
                        help='random seed')
    parser.add_argument('--bin_weight', type=float, default=5.0,
                        help='weight of binary task')
    parser.add_argument('--do_train', action='store_true',
                        help='do train')
    parser.add_argument('--do_eval', action='store_true',
                        help='do eval')
    parser.add_argument('--overwrite', action='store_true',
                        help='overwrite existing checkpoint')
    args = parser.parse_args()
    if args.do_train:
        trained_model = train(bert_model=args.bert_model,
                              pos_weight=args.pos_weight,
                              bin_weight=args.bin_weight,
                              num_relation=args.num_relation,
                              num_class=args.num_class,
                              warmup_steps=args.warmup_steps,
                              max_epoch=args.epochs,
                              weight_decay_rate=args.weight_decay_rate,
                              learning_rate=args.learning_rate,
                              seed=args.seed,
                              max_seq_len=args.seq_len,
                              hidden_state_unit=args.hidden_state_unit,
                              batch_size=args.batch_size,
                              data_dir=args.data_dir)
        trained_model.save_weights(f'saved_model/{args.task}', overwrite=args.overwrite)
    if args.do_eval:
        evaluate(bert_model=args.bert_model,
                 task=args.task,
                 num_relation=args.num_relation,
                 num_class=args.num_class,
                 max_seq_len=args.seq_len,
                 hidden_state_unit=args.hidden_state_unit,
                 data_dir=args.data_dir)

