import os
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


def split(features, labels, num_split=800):
    dev_index = random.sample([i for i in range(len(labels))], num_split)
    train_features, train_labels = {
            'input_ids': [],
            'attention_mask': [],
            'e1_mask': [],
            'e2_mask': [],
            'pos': [],
            'r_label': []
        }, []
    dev_features, dev_labels = {
            'input_ids': [],
            'attention_mask': [],
            'e1_mask': [],
            'e2_mask': [],
            'pos': [],
            'r_label': []
        }, []

    def add_feature(new_features, index):
        for key in features.keys():
            new_features[key].append(features[key][index])

    for i, label in enumerate(labels):
        if i in dev_index:
            add_feature(dev_features, i)
            dev_labels.append(label)
        else:
            add_feature(train_features, i)
            train_labels.append(label)
    return (train_features, train_labels), (dev_features, dev_labels)


def load_dataset(dataset_path, max_seq_len, num_class):
    train_file = os.path.join(dataset_path, 'train.txt')
    test_file = os.path.join(dataset_path, 'test.txt')

    def load_data(path):
        features = {
            'input_ids': [],
            'attention_mask': [],
            'e1_mask': [],
            'e2_mask': [],
            'pos': [],
            'r_label': []
        }
        labels = []

        with open(path) as fp:
            lines = [line.strip() for line in fp]

        for i in range(0, len(lines), 4):
            input_ids = np.fromstring(lines[i], dtype='int32', sep=' ')
            features['input_ids'].append(input_ids)
            features['attention_mask'].append([True] * len(input_ids))
            e1_mask = list(np.fromstring(lines[i + 1], dtype='bool', sep=' '))
            e2_mask = list(np.fromstring(lines[i + 2], dtype='bool', sep=' '))
            e11_pos = e1_mask.index(True) - 1
            e12_pos = e1_mask.index(False, e11_pos + 1)
            e21_pos = e2_mask.index(True) - 1
            e22_pos = e2_mask.index(False, e21_pos + 1)
            pos = [0, e11_pos, e12_pos, e21_pos, e22_pos, len(e1_mask) - 1]
            features['pos'].append(pos)
            features['e1_mask'].append(e1_mask)
            features['e2_mask'].append(e2_mask)
            label = int(lines[i + 3])
            l = np.fromstring(lines[i + 3], dtype='int32', sep=' ')
            features['r_label'].append(((l - 1 + num_class) % num_class) // 2)
            labels.append(np.array([label]))

        # padding
        for key in features:
            if key.endswith('pos') or key == 's_partition' or key == 'r_label':
                continue
            if key != 'srl_ids' and key != 'partitions' and key != 'e1_mask' and key != 'e2_mask':
                features[key] = pad_sequences(features[key],
                                              dtype=('bool' if key.endswith('mask') else 'int32'),
                                              padding='post',
                                              maxlen=max_seq_len)
            else:
                features[key] = pad_sequences(features[key],
                                              dtype=('bool' if key.endswith('mask') else 'int32'),
                                              padding='post',
                                              maxlen=max_seq_len)

        return features, labels

    train_data = load_data(train_file)
    test_data = load_data(test_file)
    size = len(train_data[1])
    train_data = tf.data.Dataset.from_tensor_slices(train_data)
    test_data = tf.data.Dataset.from_tensor_slices(test_data)

    return train_data, test_data, size

