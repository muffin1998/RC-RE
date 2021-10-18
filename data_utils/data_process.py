import os
from tqdm import tqdm
from transformers import AutoTokenizer
import re


def clean_str(text):
    text = text.lower()
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"that's", "that is ", text)
    text = re.sub(r"there's", "there is ", text)
    text = re.sub(r"it's", "it is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()


def list2str(arr):
    return ' '.join(str(x) for x in arr)


def process_data(sentences,
                 labels,
                 label2id,
                 tokenizer,
                 output_path):

    data = [[] for _ in range(4)]
    length = []
    for (sentence, label) in tqdm(list(zip(sentences, labels))):
        bert_tokens = tokenizer.tokenize(sentence, truncation=False)
        length.append(len(bert_tokens) + 2)
        ''' Create entity mask with CLS and SEP included '''
        ''' Encode tokens, encode result contains [CLS] and [SEP] '''
        if tokenizer.name_or_path.find('roberta') != -1:
            bert_tokens.insert(0, '<s>')
            bert_tokens.append('</s>')
        else:
            bert_tokens.insert(0, '[CLS]')
            bert_tokens.append('[SEP]')
        e1_start = bert_tokens.index('e11')
        e1_end = bert_tokens.index('e12', e1_start)
        e2_start = bert_tokens.index('e21')
        e2_end = bert_tokens.index('e22', e2_start)
        assert e1_start < e1_end and e2_start < e2_end
        if e1_start + 1 < e1_end:
            e1_start += 1
            e1_end -= 1
        else:
            e1_end -= 1
        if e2_start + 1 < e2_end:
            e2_start += 1
            e2_end -= 1
        else:
            e2_end -= 1
        e1_mask = [1 if i in range(e1_start, e1_end + 1) else 0 for i in range(len(bert_tokens))]
        e2_mask = [1 if i in range(e2_start, e2_end + 1) else 0 for i in range(len(bert_tokens))]
        ''' Replace entity mark with $ and # '''
        replace_dict = {
            'e11': '$',
            'e12': '$',
            'e21': '#',
            'e22': '#'
        }
        for (key, value) in replace_dict.items():
            bert_tokens[bert_tokens.index(key)] = value
        input_ids = tokenizer.convert_tokens_to_ids(bert_tokens)

        data[0].append(input_ids)
        data[1].append(e1_mask)
        data[2].append(e2_mask)
        if label2id is not None:
            data[3].append(label2id[label])
        else:
            data[3].append(int(label))

    with open(output_path, 'w') as fp:
        for (input_ids, e1_mask, e2_mask, label) in tqdm(list(zip(*data))):
            fp.write(list2str(input_ids) + '\n')
            fp.write(list2str(e1_mask) + '\n')
            fp.write(list2str(e2_mask) + '\n')
            fp.write(str(label) + '\n')
    length.sort()
    print(length[-10: -1])


def process_tac40(dataset_path, bert_model):
    train_file = os.path.join(dataset_path, 'raw/train.txt')
    test_file = os.path.join(dataset_path, 'raw/test.txt')
    output_path = os.path.join(dataset_path)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    ''' Init Bert Tokenizer '''
    tokenizer = AutoTokenizer.from_pretrained(bert_model, do_lower_case=True)
    entity_mark = ['e11', 'e12', 'e21', 'e22']
    tokenizer.add_special_tokens({'additional_special_tokens': entity_mark})

    replace = {
        '<e1>': 'e11',
        '</e1>': 'e12',
        '<e2>': 'e21',
        '</e2>': 'e22'
    }

    def process(data_file):
        sentences = []
        labels = []

        with open(data_file) as fp:
            lines = [line.strip() for line in fp if line.strip()]

        for i in tqdm(range(0, len(lines), 2)):
            sentence = lines[i].split('\t')[1].strip('"').strip()
            label = lines[i + 1].strip()
            sentence = sentence.strip()
            for (k, v) in replace.items():
                sentence = sentence.replace(k, v)
            sentences.append(sentence)
            labels.append(label)

        # count_tag(sentences, spacy_tokenizer, srl_tagger)
        process_data(sentences,
                     labels,
                     None,
                     tokenizer,
                     os.path.join(output_path, data_file.split('/')[-1]))

    process(train_file)
    process(test_file)


def process_kbp37(dataset_path, bert_model):
    label_file = os.path.join(dataset_path, 'labels.txt')
    train_file = os.path.join(dataset_path, 'raw/train.txt')
    dev_file = os.path.join(dataset_path, 'raw/dev.txt')
    test_file = os.path.join(dataset_path, 'raw/test.txt')
    output_path = os.path.join(dataset_path)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    ''' Init Bert Tokenizer '''
    tokenizer = AutoTokenizer.from_pretrained(bert_model, do_lower_case=True)
    entity_mark = ['e11', 'e12', 'e21', 'e22']
    tokenizer.add_special_tokens({'additional_special_tokens': entity_mark})

    label2id = {}
    with open(label_file) as f:
        for (i, label) in enumerate(f):
            label2id[label.strip()] = i

    replace = {
        '<e1>': 'e11',
        '</e1>': 'e12',
        '<e2>': 'e21',
        '</e2>': 'e22'
    }

    def process(data_file):
        sentences = []
        labels = []

        with open(data_file) as fp:
            lines = [line.strip() for line in fp if line.strip()]

        for i in tqdm(range(0, len(lines), 2)):
            sentence = lines[i].split('\t')[1].strip('"').strip()
            for (k, v) in replace.items():
                sentence = sentence.replace(k, v)
            sentences.append(sentence)
            labels.append(lines[i + 1].strip())

        # count_tag(sentences, spacy_tokenizer, srl_tagger)
        process_data(sentences,
                     labels,
                     label2id,
                     tokenizer,
                     os.path.join(output_path, data_file.split('/')[-1]))

    process(train_file)
    process(dev_file)
    process(test_file)


def process_semeval(dataset_path, bert_model):
    label_file = os.path.join(dataset_path, 'raw/labels.txt')
    train_file = os.path.join(dataset_path, 'raw/train.txt')
    test_file = os.path.join(dataset_path, 'raw/test.txt')
    output_path = os.path.join(dataset_path)

    ''' Init Bert Tokenizer '''
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_model, do_lower_case=True)
    entity_mark = ['e11', 'e12', 'e21', 'e22']
    bert_tokenizer.add_special_tokens({'additional_special_tokens': entity_mark})

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    def process(data_file):
        sentences = []
        labels = []

        with open(data_file) as fp:
            lines = [line.strip() for line in fp]

        for i in tqdm(range(0, len(lines), 2)):
            sentences.append(lines[i].strip())
            labels.append(int(lines[i + 1]))

        # count_tag(sentences, spacy_tokenizer, srl_tagger)
        process_data(sentences,
                     labels,
                     None,
                     tokenizer=bert_tokenizer,
                     output_path=os.path.join(output_path, data_file.split('/')[-1]))

    process(train_file)
    process(test_file)
