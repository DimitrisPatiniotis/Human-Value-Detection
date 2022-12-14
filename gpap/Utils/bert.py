import os
import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AdamW
from transformers import BertModel, AutoConfig

from settings import *

def F1_as_evaluator(y_pred, y_true, target_cols):
    """
    F1 computed as in https://github.com/touche-webis-de/touche-code/blob/main/semeval23/human-value-detection/evaluator/evaluator.py

    :param y_pred: numpy array with the predicted values. Shape: [total_batches * batch_size, length of target_cols]
    :param y_true: numpy array of the true values. Shape: [total_batches * batch_size, length of target_cols]
    :param target_cols: List of the selected values
    :return: fmeasure, precision, recall, fmeasures, precisions, recalls
    """

    def initializeCounter(availableValues):
        counter = {}
        for value_ in availableValues:
            counter[value_] = 0
        return counter

    relevants = initializeCounter(target_cols)
    positives = initializeCounter(target_cols)
    truePositives = initializeCounter(target_cols)

    for labels in y_true:
        for value, label in enumerate(labels):
            if label == 1:
                relevants[target_cols[value]] += 1

    for argumentId, labels in enumerate(y_pred):
        for value, label in enumerate(labels):
            if label == 1:
                positives[target_cols[value]] += 1
                if y_true[argumentId][value] == 1:
                    truePositives[target_cols[value]] += 1

    precisions = []
    recalls = []
    fmeasures = []
    for value in target_cols:
        if relevants[value] != 0:
            precision = 0
            if positives[value] != 0:
                precision = truePositives[value] / positives[value]
            precisions.append(precision)
            recall = truePositives[value] / relevants[value]
            recalls.append(recall)
            fmeasure = 0
            if precision + recall != 0:
                fmeasure = 2 * precision * recall / (precision + recall)
            fmeasures.append(fmeasure)
    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    fmeasure = 2 * precision * recall / (precision + recall)

    return fmeasure, precision, recall, fmeasures, precisions, recalls

def BCE_loss(outputs, targets, weights=None):

    return torch.nn.BCEWithLogitsLoss(weight=weights)(outputs, targets)

def f1_loss(y_pred, y_true, weights=None):
    """
    Calculate F1 score. Can work with gpu tensors

    Reference
    ---------
    - https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
    - Paper: sigmoidF1: A Smooth F1 Score Surrogate Loss for Multilabel Classification

    """
    sig_ = torch.nn.Sigmoid()
    y_pred_probs = sig_(y_pred)

    tp = (y_true * y_pred_probs).to(torch.float32)
    if weights is not None:
        # Reverse weights since sigmoidF1 will be reversed, i.e., 1-sigmoidF1 .
        tp = tp * (1.00 - weights)
    tp = tp.mean(dim=0)

    fp = ((1 - y_true) * y_pred_probs).mean(dim=0).to(torch.float32)
    fn = (y_true * (1 - y_pred_probs)).mean(dim=0).to(torch.float32)

    epsilon = 1e-7

    sigmoidF1 = (2*tp) / ((2*tp) + fn + fp + epsilon)
    mean_sigmoidF1 = sigmoidF1.mean()

    reverse_mean_sigmoidF1 = 1.00 - mean_sigmoidF1

    return reverse_mean_sigmoidF1

class BERTDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, target_cols, train=False):
        self.df = df
        self.max_len = max_len
        self.text = df.Text
        self.tokenizer = tokenizer
        self.targets = df[target_cols].values
        self.train = train

        if self.train and W_LOSS_WEIGHTS:
            # Calculations based on:
            # https://naadispeaks.wordpress.com/2021/07/31/handling-imbalanced-classes-with-weighted-loss-in-pytorch/

            if not SINGLE_CLASS:
                print('When "W_LOSS_WEIGHTS", then "SINGLE_CLASS" should be "True" but found to be "False" !!!')
                exit(0)
            else:
                num_train_samples_per_binary_class = df[target_cols].value_counts()
                self.train_loss_weights = \
                    (1 - (num_train_samples_per_binary_class/num_train_samples_per_binary_class.sum())).values

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        text = self.text[index]

        # Replace some words with '[MASK]'
        if self.train and W_MASKING:

            # Create an array of indices of the worlds to be masked and mask the 'text'
            np_of_words = np.array(list(text.split(' ')))
            masked_np_of_words = np_of_words.copy()
            mask_indices = np.random.randint(np_of_words.shape[0], size=(round(MAKING_PERC*np_of_words.shape[0])))
            mask_indices_to_be_kept = np_of_words[mask_indices] != '[SEP]'
            kept_mask_indices = mask_indices[mask_indices_to_be_kept]
            masked_np_of_words[kept_mask_indices] = '[MASK]'

            text = ''
            for word_id, word in enumerate(masked_np_of_words):
                text += word + (' ' if word_id < masked_np_of_words.shape[0] else '')

        inputs = self.tokenizer.encode_plus(text,
                                            truncation=True,
                                            add_special_tokens=True if ADD_SPECIAL_TOKENS else False,
                                            max_length=self.max_len,
                                            padding='max_length',
                                            return_token_type_ids=False)
        ids = inputs['input_ids']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

class BERTClass(torch.nn.Module):
    def __init__(self, target_cols, max_length, device='cpu', dl=None, loader_train_dataset=None, loader_valid_dataset=None):

        super(BERTClass, self).__init__()

        self.loader_train_dataset = loader_train_dataset
        self.loader_valid_dataset = loader_valid_dataset
        self.dl = dl
        self.max_length = max_length
        self.device = device
        self.freeze_bert = FREEZE_BERT
        self.head_type = HEAD_TYPE
        self.multihead = MULTIHEAD
        self.biodirectional_GRU = BIODIRECTIONAL_GRU
        self.dropout_rate = DROPOUT
        self.GRU_hidden_dim = GRU_HIDDEN_DIM
        self.target_cols = target_cols
        self.train_dataset = None
        self.valid_dataset = None

        configuration = AutoConfig.from_pretrained('bert-base-uncased')
        configuration.hidden_dropout_prob = HIDDEN_DROPOUT_PROB
        configuration.attention_probs_dropout_prob = ATTENTION_PROBS_DROPOUT_PROBS
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path='bert-base-uncased', config=configuration)

        if max_length > 512:
            self.bert.config.max_position_embeddings = max_length
            self.bert.base_model.embeddings.position_ids = torch.arange(max_length, dtype=torch.int).expand((1, -1))
            self.bert.base_model.embeddings.token_type_ids = torch.zeros(max_length, dtype=torch.int).expand((1, -1))
            orig_pos_emb = self.bert.base_model.embeddings.position_embeddings.weight
            self.bert.base_model.embeddings.position_embeddings.weight = torch.nn.Parameter(torch.cat((orig_pos_emb, orig_pos_emb)))
            print('Max len exceeds 512 tokens !!! BERT embeddings are changed !!!')

        # Freeze the parameters of BERT tokenizer
        if self.freeze_bert:
            for name, param in self.bert.named_parameters():
                if name.startswith('bert'):
                    param.requires_grad = False

        embedding_dim = self.bert.config.to_dict()['hidden_size']

        if self.head_type == 'MLP':
            if not self.multihead:
                self.fc = torch.nn.Linear(embedding_dim*(max_length
                                                         if not W_CLS_ONLY_FIX
                                                         else 1), len(self.target_cols))
            else:
                self.fcs = torch.nn.ModuleList([torch.nn.Linear(embedding_dim*(max_length
                                                                               if not W_CLS_ONLY_FIX
                                                                               else 1), 1)
                                                for _ in range(len(self.target_cols))])
        elif self.head_type == 'GRU':
            if not self.multihead:
                self.rnn = torch.nn.GRU(embedding_dim,
                                        self.GRU_hidden_dim,
                                        num_layers=1,
                                        bidirectional=self.biodirectional_GRU,
                                        batch_first=True,
                                        dropout=0)
                self.fc = torch.nn.Linear(self.GRU_hidden_dim*2 if self.biodirectional_GRU
                                                                else self.GRU_hidden_dim,
                                          len(target_cols))
            else:
                self.rnns = torch.nn.ModuleList([torch.nn.GRU(embedding_dim,
                                                 self.GRU_hidden_dim,
                                                 num_layers=1,
                                                 bidirectional=self.biodirectional_GRU,
                                                 batch_first=True,
                                                 dropout=0) for _ in range(len(self.target_cols))])
                self.fcs = torch.nn.ModuleList([torch.nn.Linear(self.GRU_hidden_dim*2 if self.biodirectional_GRU
                                                                                      else self.GRU_hidden_dim,
                                                                1)
                                                for _ in range(len(self.target_cols))])

        else:
            print("You should choose a valid 'head_type' !!!")
            exit(0)

        self.dropout = torch.nn.Dropout(self.dropout_rate)

    
    def forward(self, ids):

        # ids = [batch size, sent len]

        if self.freeze_bert:
            with torch.no_grad():
                embedded = self.bert(ids)[0]
        else:
            embedded = self.bert(ids)[0]
        # embedded = [batch size, sent len, emb dim]

        if self.head_type == 'MLP':
            flat_embedded = torch.flatten(embedded, start_dim=1) \
                            if not W_CLS_ONLY_FIX \
                            else embedded[:, 0]
            # flat_embedded = [batch size, sent len * emb dim]
            #                 if W_CLS_ONLY_FIX
            #                 else [batch size, emb dim]

            dropout = self.dropout(flat_embedded)

            if not self.multihead:
                output = self.fc(dropout)
                # output = [batch size, num classes]
            else:
                outputs = [fc(dropout) for fc in self.fcs]
                # outputs = list of 20 tensors each of size [batch size, 1]
                output = torch.concat(outputs, dim=1)
                # output = [batch size, num classes]

        elif self.head_type == 'GRU':

            if not self.multihead:
                _, hidden = self.rnn(embedded)
                # hidden = [n layers * n directions, batch size, emb dim]

                if self.biodirectional_GRU:
                    hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
                else:
                    hidden = self.dropout(hidden[-1, :, :])
                # hidden = [batch size, hid dim]

                dropout = self.dropout(hidden)

                output = self.fc(dropout)
                # output = [batch size, num classes]

            else:
                hiddens = [rnn(embedded)[1] for rnn in self.rnns]
                # hiddens = list of 20 tensors each of size [n layers * n directions, batch size, emb dim]

                if self.biodirectional_GRU:
                    hiddens = [torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1) for hidden in hiddens]
                else:
                    hiddens = [self.dropout(hidden[-1, :, :]) for hidden in hiddens]
                # hiddens = list of 20 tensors each of size [batch size, hid dim]

                dropouts = [self.dropout(hidden) for hidden in hiddens]

                outputs = [fc(dropout) for (dropout, fc) in zip(dropouts, self.fcs)]
                # outputs = list of 20 tensors each of size [batch size, 1]
                output = torch.concat(outputs, dim=1)
                # output = [batch size, num classes]

        return output

    def one_epoch_train(self, epoch, tloader, vloader, device, optimizer):

        ######## Train the model ########
        self.train()

        loss_list = []
        for _, data in tqdm(enumerate(tloader, 0)):
            ids = data['ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)

            outputs = self(ids)

            if W_LOSS_WEIGHTS:
                loss_weights_repeated_batch_size = np.repeat(self.train_dataset.train_loss_weights[np.newaxis, :], outputs.size()[0], axis=0)
                final_loss_weights = np.take_along_axis(loss_weights_repeated_batch_size, np.array(targets.cpu().numpy(), dtype=np.int), axis=1)

            loss = BCE_loss(outputs, targets, None if not W_LOSS_WEIGHTS
                                                   else torch.from_numpy(final_loss_weights).to(device, dtype=torch.float32)) \
                   if LOSS == 'BCE' \
                   else (f1_loss(outputs, targets, None if not W_LOSS_WEIGHTS
                                                        else torch.from_numpy(final_loss_weights).to(device, dtype=torch.float32))
                         if LOSS == 'sigmoidF1'
                         else 0.0)

            loss_list.append(loss if LOSS == 'sigmoidF1' else loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        ######## Evaluate the model ########
        print('Evaluating...')
        self.eval()

        v_loss_list = []
        v_pred_one_hot = []
        v_ground_truth = []
        for _, v_data in tqdm(enumerate(vloader, 0)):
            v_ids = v_data['ids'].to(device, dtype=torch.long)
            v_targets = v_data['targets'].to(device, dtype=torch.float)

            with torch.no_grad():
                v_outputs = self(v_ids)

                if W_LOSS_WEIGHTS:
                    v_loss_weights_repeated_batch_size = np.repeat(self.train_dataset.train_loss_weights[np.newaxis, :],
                                                                   v_outputs.size()[0], axis=0)
                    v_final_loss_weights = np.take_along_axis(v_loss_weights_repeated_batch_size,
                                                            np.array(v_targets.cpu().numpy(), dtype=np.int), axis=1)

                v_loss = BCE_loss(v_outputs, v_targets, None if not W_LOSS_WEIGHTS
                                                             else torch.from_numpy(v_final_loss_weights).to(device, dtype=torch.float32)) \
                         if LOSS == 'BCE' \
                         else (f1_loss(v_outputs, v_targets, None if not W_LOSS_WEIGHTS
                                                                  else torch.from_numpy(v_final_loss_weights).to(device, dtype=torch.float32))
                               if LOSS == 'sigmoidF1'
                               else 0.0)

                v_loss_list.append(v_loss if LOSS == 'sigmoidF1' else v_loss.item())

                sig = torch.nn.Sigmoid()
                v_probs = sig(v_outputs)

            v_pred_one_hot.append(np.around(v_probs.cpu().numpy()))
            v_ground_truth.append(v_targets.cpu().numpy())

        v_conc_pred_one_hot = np.concatenate(v_pred_one_hot, axis=0)
        v_conc_ground_truth = np.concatenate(v_ground_truth, axis=0)
        f1_micro_average = f1_score(y_true=v_conc_ground_truth, y_pred=v_conc_pred_one_hot, average='micro', zero_division=0)
        f, pr, rec, fs, prs, recs = F1_as_evaluator(v_conc_pred_one_hot, v_conc_ground_truth, self.target_cols)

        print(f'Epoch: {epoch}, Loss:  {sum(loss_list) / len(loss_list):.2f}, '
              f'Val Loss: {sum(v_loss_list) / len(v_loss_list):.2f}, Val Micro F1: {f1_micro_average:.2f}, '
              f'Val evaluator F1: {f: .2f}')

        return sum(v_loss_list) / len(v_loss_list), f1_micro_average

    def best_model_evaluation(self, vloader, device, evaluation_only=False):

        if not evaluation_only:
            print('Training has ended. Loading best model for evaluation..')
        else:
            print('Loading saved model for evaluation..')

        self.load_state_dict(torch.load(MODEL_PATH if not evaluation_only else MODEL_PATH_FOR_EVALUATION_ONLY,
                                        map_location=torch.device(device)))
        self.eval()

        final_v_loss_list = []
        final_v_pred_one_hot = []
        final_v_ground_truth = []
        for _, v_data in tqdm(enumerate(vloader, 0)):
            v_ids = v_data['ids'].to(device, dtype=torch.long)
            v_targets = v_data['targets'].to(device, dtype=torch.float)

            with torch.no_grad():
                v_outputs = self(v_ids)

                if W_LOSS_WEIGHTS:
                    v_loss_weights_repeated_batch_size = np.repeat(self.train_dataset.train_loss_weights[np.newaxis, :],
                                                                   v_outputs.size()[0], axis=0)
                    v_final_loss_weights = np.take_along_axis(v_loss_weights_repeated_batch_size,
                                                              np.array(v_targets.cpu().numpy(), dtype=np.int), axis=1)

                v_loss = BCE_loss(v_outputs, v_targets, None if not W_LOSS_WEIGHTS
                                                             else torch.from_numpy(v_final_loss_weights).to(device, dtype=torch.float32)) \
                         if LOSS == 'BCE' \
                         else (f1_loss(v_outputs, v_targets, None if not W_LOSS_WEIGHTS
                                                                  else torch.from_numpy(v_final_loss_weights).to(device, dtype=torch.float32))
                               if LOSS == 'sigmoidF1'
                               else 0.0)

                final_v_loss_list.append(v_loss if LOSS == 'sigmoidF1' else v_loss.item())

                sig = torch.nn.Sigmoid()
                v_probs = sig(v_outputs)

            final_v_pred_one_hot.append(np.around(v_probs.cpu().numpy()))
            final_v_ground_truth.append(v_targets.cpu().numpy())

        v_conc_pred_one_hot = np.concatenate(final_v_pred_one_hot, axis=0)
        v_conc_ground_truth = np.concatenate(final_v_ground_truth, axis=0)
        final_f1_micro_average = f1_score(y_true=v_conc_ground_truth,
                                          y_pred=v_conc_pred_one_hot,
                                          average='micro',
                                          zero_division=0)
        v_clr_dict = classification_report(v_conc_ground_truth,
                                           v_conc_pred_one_hot,
                                           zero_division=0)
        final_f, final_pr, final_rec, final_fs, final_prs, final_recs = F1_as_evaluator(v_conc_pred_one_hot, v_conc_ground_truth, self.target_cols)

        print(f'Best model: \n'
              f'Val Loss: {sum(final_v_loss_list) / len(final_v_loss_list):.2f}, Val micro F1: {final_f1_micro_average:.2f}\n'
              f'Classification Report: \n {v_clr_dict}\n'
              f' Val evaluator F1: {final_f:.2f}, Val evaluator Precision: {final_pr:.2f}, Val evaluator Recall: {final_rec:.2f}\n')

        for value_col_id, value_col in enumerate(self.target_cols):
            print("measure {\n key: \"Precision " + value_col + "\"\n value: \"" + str(final_prs[value_col_id]) + "\"\n}\n" +
                  "measure {\n key: \"Recall " + value_col + "\"\n value: \"" + str(final_recs[value_col_id]) + "\"\n}\n" +
                  "measure {\n key: \"F1 " + value_col + "\"\n value: \"" + str(final_fs[value_col_id]) + "\"\n}\n")

    def train_(self):

        #Create a directory to store the model
        if not os.path.exists('/'.join(MODEL_PATH.split('/')[:-1])):
            os.mkdir('/'.join(MODEL_PATH.split('/')[:-1]))

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        if self.max_length > 512:
            tokenizer.model_max_length = self.max_length
            print('Max len exceeds 512 tokens !!! Tokenizer is changed !!!')

        self.to(device)

        optimizer = AdamW(params=self.parameters(), lr=LEARNING_RATE, weight_decay=1e-6, no_deprecation_warning=True)

        if not W_NEW_DATA:
            train_df_for_BERTDataset = self.dl.train
            validation_df_for_BERTDataset = self.dl.validation
        else:
            train_df_for_BERTDataset = self.loader_train_dataset.workingTable
            validation_df_for_BERTDataset = self.loader_valid_dataset.workingTable

        self.train_dataset = BERTDataset(train_df_for_BERTDataset, tokenizer, self.max_length,
                                         target_cols=self.target_cols, train=True)
        self.valid_dataset = BERTDataset(validation_df_for_BERTDataset, tokenizer, self.max_length,
                                         target_cols=self.target_cols)

        train_loader = DataLoader(self.train_dataset, batch_size=TRAIN_BATCH_SIZE,
                                  num_workers=4, shuffle=True, pin_memory=True)
        valid_loader = DataLoader(self.valid_dataset, batch_size=TRAIN_BATCH_SIZE,
                                  num_workers=4, shuffle=False, pin_memory=True)

        print('Starting training...')
        best_val_loss = np.inf
        epochs_wo_improve = 0
        for epoch in range(EPOCHS):
            val_loss, val_F1 = self.one_epoch_train(epoch, train_loader, valid_loader, device, optimizer)

            if best_val_loss > val_loss:
                best_val_loss = val_loss
                epochs_wo_improve = 0
                torch.save(self.state_dict(), MODEL_PATH)
            elif epochs_wo_improve > PATIENCE:
                print(f'Early stopping at epoch {epoch} !')
                break
            else:
                epochs_wo_improve += 1

        self.best_model_evaluation(valid_loader, device)

    def evaluate_(self):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        if self.max_length > 512:
            tokenizer.model_max_length = self.max_length
            print('Max len exceeds 512 tokens !!! Tokenizer is changed !!!')

        self.to(device)

        if not W_NEW_DATA:
            validation_df_for_BERTDataset = self.dl.validation
        else:
            validation_df_for_BERTDataset = self.loader_valid_dataset.workingTable

        self.valid_dataset = BERTDataset(validation_df_for_BERTDataset, tokenizer,
                                         self.max_length, target_cols=self.target_cols)
        valid_loader = DataLoader(self.valid_dataset, batch_size=TRAIN_BATCH_SIZE,
                                  num_workers=4, shuffle=False, pin_memory=True)

        self.best_model_evaluation(valid_loader, device, evaluation_only=True)
    
if __name__ == '__main__':
    print('BERT Utils')