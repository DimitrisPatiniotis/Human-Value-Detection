import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
import torch
from torch import nn
import transformers
from transformers import AutoModel, BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

def get_labels():
    with open('../Data/value-categories.json') as jsonFile:
        value_categories_json = json.load(jsonFile)
    l = list(value_categories_json.keys())
    return l

def load_dataset(selected_label):
    sep = '\t'
    with open('../Data/value-categories.json') as jsonFile:
        value_categories_json = json.load(jsonFile)
    label_names = list(value_categories_json.keys())
    X_train = pd.read_csv('../Data/arguments-training.tsv', sep=sep, on_bad_lines='skip')
    y_train = pd.read_csv('../Data/labels-training.tsv', sep=sep, on_bad_lines='skip')
    X_val = pd.read_csv('../Data/arguments-validation.tsv', sep=sep, on_bad_lines='skip')
    y_val = pd.read_csv('../Data/labels-validation.tsv', sep=sep, on_bad_lines='skip')

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


# Tokenization
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


class BERT(nn.Module):
    def __init__(self, bert):
        super(BERT, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 2)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
        out = self.fc1(cls_hs)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.softmax(out)

        return out


def train_for_label(device, label):
    X_train, y_train, X_val, y_val, label_names = load_dataset(label)

    PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
    MAX_SEQUENCE_LEN = 500
    BATCH_SIZE = 16
    EPOCHS = 10
    LEARNING_RATE = 0.001
    bert = AutoModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    # Freeze bert params
    for param in bert.parameters():
        param.requires_grad = False
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    # Tokenize Text
    train_inputs, train_masks = preprocessing(
        X_train, tokenizer, MAX_SEQUENCE_LEN)
    val_inputs, val_masks = preprocessing(X_val, tokenizer, MAX_SEQUENCE_LEN)

    # Prepair DataLoaders
    train_labels, val_labels = torch.tensor(y_train), torch.tensor(y_val)

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

    # Validation
    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=BATCH_SIZE)

    # # Compute Weight of Classes
    train_class_weights = compute_class_weight(class_weight = 'balanced', classes = np.unique(y_train), y = y_train)
    # print('CLASS WEIGHTS', train_class_weights)
    train_class_weights= torch.tensor(train_class_weights,dtype=torch.float)
    train_class_weights = train_class_weights.to(device)
    # Define Model
    model = BERT(bert)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    cross_entropy  = nn.NLLLoss(weight=train_class_weights) 


    def train():
        model.train()

        total_loss, total_accuracy = 0, 0

        total_preds = []
        total_labels = []

        for step, batch in enumerate(train_dataloader):
            if step % 50 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(
                    step, len(train_dataloader)))

            # push the batch to gpu
            batch = [r.to(device) for r in batch]

            sent_id, mask, labels = batch
            model.zero_grad()
            preds = model(sent_id, mask)

            loss = cross_entropy(preds, labels)
            total_loss = total_loss + loss.item()
            loss.backward()
            optimizer.step()


            preds = preds.detach().cpu().numpy()
            preds = np.argmax(preds, axis=1)
            total_preds += list(preds)
            total_labels += labels.tolist()

        avg_loss = total_loss / len(train_dataloader)


        f1 = f1_score(total_labels, total_preds, average='weighted')
        return avg_loss, f1, model

    def evaluate():

        print("\nEvaluating...")

        # deactivate dropout layers
        model.eval()

        total_loss, total_accuracy = 0, 0

        total_preds = []
        total_labels = []

        for step, batch in enumerate(val_dataloader):

            # Progress update every 50 batches.
            if step % 50 == 0 and not step == 0:

                print('  Batch {:>5,}  of  {:>5,}.'.format(
                    step, len(val_dataloader)))

            batch = [t.to(device) for t in batch]

            sent_id, mask, labels = batch

            with torch.no_grad():

                preds = model(sent_id, mask)
                loss = cross_entropy(preds, labels)
                total_loss = total_loss + loss.item()

                preds = preds.detach().cpu().numpy()
                preds = np.argmax(preds, axis=1)
                total_preds += list(preds)
                total_labels += labels.tolist()

        avg_loss = total_loss / len(val_dataloader)

        f1 = f1_score(total_labels, total_preds, average='weighted')
        return avg_loss, f1

    best_valid_loss = float('inf')

    train_losses = []
    valid_losses = []

    for epoch in range(EPOCHS):

        print('\n {} Epoch {:} / {:}'.format(label, epoch + 1, EPOCHS))

        train_loss, f1_train, model = train()

        valid_loss, f1_valid = evaluate()

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')
        print(f'\nTraining F1: {f1_train:.3f}')
        print(f'Validation F1: {f1_valid:.3f}')
    
    # Save model
    torch.save(model, 'models/{}.pt'.format(label))



def models_train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    labels = get_labels()

    # Train
    for label in labels:
        train_for_label(device, label)

    return
    

def models_predict():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    labels = get_labels()

    # Predict

    return 


if __name__ == '__main__':
    models_train()