import json
import numpy as np
import random
import torch
import pandas as pd

def get_labels():
    with open('../../Data/value-categories.json') as jsonFile:
        value_categories_json = json.load(jsonFile)
    label_names = list(value_categories_json.keys())
    return label_names

def get_column(file_name, col_name, sep='\t'):
    try:
        df = pd.read_csv(file_name, sep=sep, on_bad_lines='skip')
        return df[col_name]
    except:
        print(f'Could Not Read {col_name} column from {file_name}')

def load_dataset(selected_label):
    sep = '\t'
    with open('../Data/value-categories.json') as jsonFile:
        value_categories_json = json.load(jsonFile)
    label_names = list(value_categories_json.keys())
    X_train = pd.read_csv('../Data/arguments-training.tsv',
                          sep=sep, on_bad_lines='skip')
    y_train = pd.read_csv('../Data/labels-training.tsv',
                          sep=sep, on_bad_lines='skip')
    X_val = pd.read_csv('../Data/arguments-validation.tsv',
                        sep=sep, on_bad_lines='skip')
    y_val = pd.read_csv('../Data/labels-validation.tsv',
                        sep=sep, on_bad_lines='skip')

    # Concat X Columns
    X_train['Stance'], X_val['Stance'] = X_train['Stance'].apply(
        lambda txt: ' '+txt+' '), X_val['Stance'].apply(lambda txt: ' '+txt+' ')
    X_train['Text'] = X_train['Conclusion'] + \
        X_train['Stance'] + X_train['Premise']
    X_val['Text'] = X_val['Conclusion'] + X_val['Stance'] + X_val['Premise']

    X_train = X_train.Text.values
    X_val = X_val.Text.values

    # Select Target Label
    if selected_label in label_names:
        y_train = y_train[selected_label]
        y_val = y_val[selected_label]
    else:
        print('Please select a valid label')

    # Check max Len
    print('Max training length is {}'.format(len(max(list(X_train), key=len))))
    # Todo: print histogram of len distribution
    seq_len = [len(i.split()) for i in X_train]
    print('Max validation length is {}'.format(len(max(list(X_val), key=len))))

    # Check label distribution
    print('Training label distribution of {} is {} (0) and {} (1)'.format(selected_label, round(
        y_train.value_counts(normalize=True)[0], 3), round(y_train.value_counts(normalize=True)[1], 3)))
    print('Validation label distribution of {} is {} (0) and {} (1)'.format(selected_label, round(
        y_val.value_counts(normalize=True)[0], 3), round(y_val.value_counts(normalize=True)[1], 3)))

    return X_train, y_train, X_val, y_val, label_names

def setSeeds(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def unimodal_concat(items):
    r = items[0]
    for i in items[1:]:
        r += f'[SEP]{i}'
    return r

def preprocessing(data, tokenizer, maxlength):

    input_ids = []
    attention_masks = []
    for argument in data:
        encoded_sent = tokenizer.encode_plus(text=argument, add_special_tokens=True, max_length=maxlength, pad_to_max_length=True, return_attention_mask=True)
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks