import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.utils.class_weight import compute_class_weight
from nltk.tokenize import word_tokenize
from sklearn.metrics import f1_score, classification_report
import torch
from transformers import AutoModel, BertTokenizer, BertModel
from finetune_bert import BERTModel, MAX_SEQUENCE_LEN, MODEL_SAVE_PATH, y_TRAIN_TSV, y_VAL_TSV, X_TRAIN_TSV, X_VAL_TSV, SequentialSampler, BATCH_SIZE, PRE_TRAINED_MODEL_NAME, get_x_and_y, TensorDataset, RandomSampler, DataLoader
from utils import get_labels, preprocessing, setSeeds
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names
from torchsummary import summary

import csv



def get_labels():
    with open('../../Data/value-categories.json') as jsonFile:
        value_categories_json = json.load(jsonFile)
    label_names = list(value_categories_json.keys())
    return label_names

def load_dataset(selected_label):
    sep = '\t'
    with open('../../Data/value-categories.json') as jsonFile:
        value_categories_json = json.load(jsonFile)
    label_names = list(value_categories_json.keys())
    X_train = pd.read_csv('../../Data/arguments-training.tsv', sep=sep, on_bad_lines='skip')
    y_train = pd.read_csv('../../Data/labels-training.tsv', sep=sep, on_bad_lines='skip')
    X_val = pd.read_csv('../../Data/arguments-validation.tsv', sep=sep, on_bad_lines='skip')
    y_val = pd.read_csv('../../Data/labels-validation.tsv', sep=sep, on_bad_lines='skip')

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

def PCA_reduce_dim(X, n_components=100):
    clf = decomposition.PCA(n_components=n_components)
    return clf.fit_transform(X.reshape(-1,768))


class BertWordEmbedding():

    def __init__(self, tokenizer, device):
        self.tokens = []
        self.embeddings = []
        self.tokenizer = tokenizer
        self.device = device
        print(torch.cuda.is_available())

    def set_tokens(self, token_ids):

        self.tokens = [] 
        vocab = list(self.tokenizer.vocab.keys())
        for token_id in token_ids:
            self.tokens.append(vocab[token_id])

    def get_text_embedding(self, text):

        # Reset the global variables
        self.tokens = []
        self.embeddings = []

        self.tokens = word_tokenize(text)
        # print(self.tokens)
        # print(len(self.tokens))

        if len(self.tokens) == 0:
            return None, None

        encoded = self.tokenizer.encode_plus( 
            text=self.tokens,
            add_special_tokens=True,
            is_split_into_words=True
        )
        self.set_tokens(encoded['input_ids']) 

        input_ids_tensor = torch.tensor([encoded['input_ids']]).to(self.device)
        attention_mask_tensors = torch.tensor([encoded['attention_mask']]).to(self.device)

        model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True).to(self.device)
        model.eval()

        with torch.no_grad():

            outputs = model(input_ids_tensor, attention_mask_tensors)
            hidden_states = outputs[2]
            batch_index = 0

            for token_index in range(len(self.tokens)):
                layers = []
                for layer_index in range(-4, 0):  # use last four layers
                    layers.append(hidden_states[layer_index][batch_index][token_index].tolist())

                self.embeddings.append(list(map(lambda x: x / 4, np.sum(layers, axis=0))))  # add the avg of the last four layers
            # print(self.embeddings)
        return self.tokens, self.embeddings

class Trainer():
    
    def __init__(self, X_train, y_train, X_val, y_val, class_weights, l_name, PCA=True):
        self.name = l_name
        self.PCA = PCA
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.class_weights = class_weights
        self.kernels = ['linear', 'poly', 'rbf', 'sigmoid']

    def apply_pca(self):
        pass

    def run_adaboost(self):
        model = AdaBoostClassifier()
        model.fit(self.X_train, self.y_train)
        
        model_accuracy = model.score(self.X_val, self.y_val)

        model_f1 = f1_score(self.y_val, model.predict(self.X_val))
        cl_report = classification_report(self.y_val, model.predict(self.X_val))
        print(cl_report)
        print("Adaboost accuracy: {}".format( model_accuracy ))
        print("Adaboost f1 score: {}".format( model_f1 ))
        return model_accuracy, model_f1

    def train_SVM(self, krnl):
        # Class Weights to be added
        model = svm.SVC(kernel=krnl, max_iter=10000)

        model.fit(self.X_train, self.y_train)
        model_accuracy = model.score(self.X_val, self.y_val)
        model_f1 = f1_score(list(self.y_val), list(model.predict(self.X_val)))
        cl_report = classification_report(self.y_val, model.predict(self.X_val))
        print(cl_report)
        return model_accuracy, model_f1
    
    def test_kernels(self):
        # self.apply_pca()
        for kernel in self.kernels:
            kernel_accuracy, kernel_f1 = self.train_SVM(kernel)
            print("{} - Kernel {} accuracy: {}".format(self.name, kernel, kernel_accuracy ))
            print("{} - Kernel {} f1 score: {}".format(self.name, kernel, kernel_f1 ))

def get_embeddings(finetuned, labels, tokenizer, device):
    if finetuned:
        bert = AutoModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        finetuned_bert = BERTModel(bert)
        finetuned_bert.load_state_dict(torch.load(MODEL_SAVE_PATH))
        finetuned_bert.to(device)

        finetuned_bert.eval()
        with torch.no_grad():
            # X_train, _, _, _, _ = load_dataset(labels[0])

            X_train, y_train = get_x_and_y(X_TRAIN_TSV, y_TRAIN_TSV, False)
            X_val, y_val = get_x_and_y(X_VAL_TSV, y_VAL_TSV, False)


            tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
            train_inputs, train_masks = preprocessing(X_train['Premise'], tokenizer, MAX_SEQUENCE_LEN)
            val_inputs, val_masks = preprocessing(X_val['Premise'], tokenizer, MAX_SEQUENCE_LEN)
            
            train_labels = torch.tensor(y_train)
            val_labels = torch.tensor(y_val)

            train_data = TensorDataset(train_inputs, train_masks, train_labels)
            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

            val_data = TensorDataset(val_inputs, val_masks, val_labels)
            val_sampler = SequentialSampler(val_data)
            val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=BATCH_SIZE)

            train_labels = torch.tensor(y_train)

            train_inputs, train_masks = preprocessing(X_train['Premise'], tokenizer, MAX_SEQUENCE_LEN)
            train_data = TensorDataset(train_inputs, train_masks, train_labels)
            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

            train_outs = []
            valid_outs = []
            for step, batch in enumerate(train_dataloader):
                if step % 50 == 0 and not step == 0:
                    print('Extracting Training Embeddings batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
                batch = [t.to(device) for t in batch]
                sent_id, mask, labels = batch
                with torch.no_grad():
                    preds = finetuned_bert(sent_id, mask)[1]
                    train_outs.append(preds)

            for step, batch in enumerate(val_dataloader):
                if step % 50 == 0 and not step == 0:
                    print('Extracting Validation Embeddings batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))
                batch = [t.to(device) for t in batch]
                sent_id, mask, labels = batch
                with torch.no_grad():
                    preds = finetuned_bert(sent_id, mask)[1]
                    valid_outs.append(preds)
            train_flat_outs = [item for sublist in train_outs for item in sublist]
            valid_flat_outs = [item for sublist in valid_outs for item in sublist]
            return [i.cpu() for i in train_flat_outs], [i.cpu() for i in valid_flat_outs]
                
    else:
        embedding = BertWordEmbedding(tokenizer, device)
        X_train_embeddings, X_val_embeddings = [], []

        if not os.path.exists('embeddings/X_train_embeddings.csv'):
            with open('embeddings/X_train_embeddings.csv', 'w', encoding='UTF8', newline='') as f:
                X_train, _, _, _, _ = load_dataset(labels[0])
                writer = csv.writer(f)
                header = ['id', 'embeddings']
                writer.writerow(header)
                for x in range(len(X_train)):
                    print('step {} out of {}'.format(x, len(X_train)))
                    _, x_em = embedding.get_text_embedding(X_train[x])
                    X_train_embeddings.append(list(np.sum(np.array(x_em[1:-1]), axis=0)))
                    writer.writerow([str(x+1), X_train_embeddings[-1]])
        else:
            X_train_embeddings_df = pd.read_csv('embeddings/X_train_embeddings.csv', sep=',', on_bad_lines='skip')
            X_train_embeddings = list([i.strip('][').split(', ') for i in X_train_embeddings_df['embeddings']])

        if not os.path.exists('embeddings/X_val_embeddings.csv'):
            with open('embeddings/X_val_embeddings.csv', 'w', encoding='UTF8', newline='') as f:
                _, _, X_val, _, _ = load_dataset(labels[0])
                writer = csv.writer(f)
                header = ['id', 'embeddings']
                writer.writerow(header)
                for x in range(len(X_val)):
                    print('step {} out of {}'.format(x, len(X_val)))
                    _, x_em = embedding.get_text_embedding(X_val[x])
                    X_val_embeddings.append(list(np.sum(np.array(x_em[1:-1]), axis=0)))
                    writer.writerow([str(x+1), X_val_embeddings[-1]])
        else:
            X_val_embeddings_df = pd.read_csv('embeddings/X_val_embeddings.csv', sep=',', on_bad_lines='skip')
            X_val_embeddings = list([i.strip('][').split(', ') for i in X_val_embeddings_df['embeddings']])
        
    return X_train_embeddings, X_val_embeddings

def train_svms(finetuned=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    labels = get_labels()
    setSeeds()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')
    X_train_embeddings, X_val_embeddings = get_embeddings(finetuned, labels, tokenizer, device)
    if finetuned:
        X_train_embeddings = [list(i) for i in X_train_embeddings]
        X_val_embeddings = [list(i) for i in X_val_embeddings]

    for label in labels:
        X_train, y_train, X_val, y_val, label_names = load_dataset(label)
        class_w = compute_class_weight(class_weight = 'balanced', classes = np.unique(y_train), y = y_train)

        trainer = Trainer(X_train_embeddings, y_train, X_val_embeddings, y_val, class_w, label)
        # print(type(list(X_train_embeddings[0])))
        # print(X_train_embeddings[0].cpu())
        # trainer.test_kernels()
        trainer.run_random_forest()


if __name__ == '__main__':
    train_svms(finetuned=False)