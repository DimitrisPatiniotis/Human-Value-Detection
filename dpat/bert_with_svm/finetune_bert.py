import json
import random
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoModel, BertTokenizer
from utils import get_labels, preprocessing, get_column, setSeeds, unimodal_concat
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


X_TRAIN_TSV = '../../Data/arguments-training.tsv'
y_TRAIN_TSV = '../../Data/labels-training.tsv'
X_VAL_TSV = '../../Data/arguments-validation.tsv'
y_VAL_TSV = '../../Data/labels-validation.tsv'
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
MAX_SEQUENCE_LEN = 512
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 0.001
MAX_SEQUENCE_LEN = 512
PATIENCE = 4
MODEL_SAVE_PATH = 'models/bert_model.pt'

# X -> Premise with unimodal concatenated human value categories
# y -> Stance
def get_x_and_y(premise_stance_path, hum_value_cat_path, hum_val_concat=True):
    premises = get_column(premise_stance_path, 'Premise')
    if hum_val_concat:
        human_value_cats = {}
        for i in get_labels():
            human_value_cats[i] = get_column(hum_value_cat_path, i)
        human_value_df = pd.DataFrame(human_value_cats)
        X_train_multimodal = pd.concat([premises, human_value_df], axis=1, join="inner")
        unimodal_sentences = []
        for row_num in range(len(X_train_multimodal)):
            unimodal_sentences.append(unimodal_concat(X_train_multimodal.loc[row_num, :].values.flatten().tolist()))
        X_train_unimodal = pd.DataFrame(unimodal_sentences)
    else:
        X_train_unimodal =  pd.DataFrame(premises)
    y_train = get_column(premise_stance_path, 'Stance')
    y_train = y_train.replace({"in favor of": 1}, regex=False).replace({"in favour of": 1}, regex=False).replace({'against': 0}, regex=True)
    X_train, y_train = shuffle(X_train_unimodal, y_train)
    # To avoid Series with dtype of object 
    # y_train = list(y_train)
    # print(y_train[:5])
    return X_train, y_train


class BERTModel(nn.Module):
    def __init__(self, bert, inter_dim=100):
        super(BERTModel, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, inter_dim)
        self.fc3 = nn.Linear(inter_dim, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
        out = self.fc1(cls_hs)
        out = self.relu(out)
        out = self.dropout(out)
        out_inter = self.fc2(out)
        out = self.fc3(out_inter)
        out = self.softmax(out)
        return [out, out_inter]
    

def train_bert():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    setSeeds()
    X_train, y_train = get_x_and_y(X_TRAIN_TSV, y_TRAIN_TSV)
    X_val, y_val = get_x_and_y(X_VAL_TSV, y_VAL_TSV)

    # print(len([i for i in X_train[0] if len(i)<512]))

    bert = AutoModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    print(X_train, X_val)
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    train_inputs, train_masks = preprocessing(X_train[0], tokenizer, MAX_SEQUENCE_LEN)
    val_inputs, val_masks = preprocessing(X_val[0], tokenizer, MAX_SEQUENCE_LEN)


    train_labels = torch.tensor(y_train)
    val_labels = torch.tensor(y_val)

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=BATCH_SIZE)

    train_class_weights = compute_class_weight(class_weight = 'balanced', classes = np.unique(y_train), y = y_train)
    train_class_weights= torch.tensor(train_class_weights,dtype=torch.float)
    train_class_weights = train_class_weights.to(device)


    model = BERTModel(bert)
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
            preds = model(sent_id, mask)[0]

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

                preds = model(sent_id, mask)[0]
                # print(preds)
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
    train_f1s = []
    valid_losses = []
    valid_f1s = []

    for epoch in range(EPOCHS):

        print('\n Epoch {:} / {:}'.format(epoch + 1, EPOCHS))

        train_loss, f1_train, m = train()

        valid_loss, f1_valid = evaluate()

        train_losses.append(train_loss)
        train_f1s.append(f1_train)
        valid_losses.append(valid_loss)
        valid_f1s.append(f1_valid)

        print(f'\nTraining Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')
        print(f'\nTraining F1: {f1_train:.3f}')
        print(f'Validation F1: {f1_valid:.3f}')

        if best_valid_loss > valid_loss:
            best_valid_loss = valid_loss
            epochs_wo_improve = 0
            torch.save(m.state_dict(), MODEL_SAVE_PATH)
        elif epochs_wo_improve > PATIENCE:
            print(f'Early stopping at epoch {epoch}')
            break
        else:
            epochs_wo_improve += 1
        

if __name__ == '__main__':
    train_bert()