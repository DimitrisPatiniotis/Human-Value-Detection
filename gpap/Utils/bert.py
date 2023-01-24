import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AdamW
from transformers import BertModel, AutoConfig

from settings import *
from losses import *
from dataset_constructor import BERTDataset
from write_and_evaluate_file import writeRun, evaluateRun


class BERTClass(torch.nn.Module):
    def __init__(self, target_cols, max_length, device='cpu', dl=None,
                 loader_train_dataset=None, loader_valid_dataset=None,
                 loader_test_dataset=None):

        super(BERTClass, self).__init__()

        self.loader_train_dataset = loader_train_dataset
        self.loader_valid_dataset = loader_valid_dataset
        self.loader_test_dataset = loader_test_dataset
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
        self.test_dataset = None

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

        self.embedding_dim = self.bert.config.to_dict()['hidden_size']

        if not ONLY_BERT_EMBEDDINGS:
            if self.head_type == 'MLP':
                if not self.multihead:
                    self.fc = torch.nn.Linear(self.embedding_dim*(max_length
                                                                  if not W_CLS_ONLY_FIX
                                                                  else 1), len(self.target_cols))
                else:
                    self.fcs = torch.nn.ModuleList([torch.nn.Linear(self.embedding_dim*(max_length
                                                                                        if not W_CLS_ONLY_FIX
                                                                                        else 1), 1)
                                                    for _ in range(len(self.target_cols))])
            elif self.head_type == 'GRU':
                if not self.multihead:
                    self.rnn = torch.nn.GRU(self.embedding_dim,
                                            self.GRU_hidden_dim,
                                            num_layers=1,
                                            bidirectional=self.biodirectional_GRU,
                                            batch_first=True,
                                            dropout=0)
                    self.fc = torch.nn.Linear(self.GRU_hidden_dim*2 if self.biodirectional_GRU
                                                                    else self.GRU_hidden_dim,
                                              len(target_cols))
                else:
                    self.rnns = torch.nn.ModuleList([torch.nn.GRU(self.embedding_dim,
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

        if ONLY_BERT_EMBEDDINGS:
            output = embedded[:, 0]

        else:
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

    def one_epoch_train(self, epoch, tloader, vloader, optimizer):

        ######## Train the model ########
        self.train()

        loss_list = []
        for _, data in tqdm(enumerate(tloader, 0)):
            ids = data['ids'].to(self.device, dtype=torch.long)
            targets = data['targets'].to(self.device, dtype=torch.float)

            outputs = self(ids)

            if W_LOSS_WEIGHTS:
                loss_weights_repeated_batch_size = \
                    np.repeat(self.train_dataset.train_loss_weights[np.newaxis, :], outputs.size()[0], axis=0)

                final_loss_weights = \
                    np.take_along_axis(loss_weights_repeated_batch_size,
                                       np.array(targets.cpu().numpy() if targets.cpu().numpy().shape[1] == 1
                                                                      else
                                                np.expand_dims(targets.cpu().numpy(), axis=2),
                                                dtype=int),
                                       axis=1 if targets.cpu().numpy().shape[1] == 1 else 2)

                if targets.cpu().numpy().shape[1] > 1:
                    final_loss_weights = np.squeeze(final_loss_weights, axis=2)

            loss = BCE_loss(outputs, targets,
                            None if not W_LOSS_WEIGHTS
                                 else torch.from_numpy(final_loss_weights).to(self.device, dtype=torch.float32)) \
                   if LOSS == 'BCE' \
                   else (f1_loss(outputs, targets,
                                 None if not W_LOSS_WEIGHTS
                                      else torch.from_numpy(final_loss_weights).to(self.device, dtype=torch.float32))
                         if LOSS == 'sigmoidF1'
                         else 0.0)

            loss_list.append(loss if LOSS == 'sigmoidF1' else loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        ######## Evaluate the model ########
        print('Evaluating...')

        v_conc_pred_one_hot, v_conc_ground_truth, v_loss_list = self.evaluation(vloader)

        f1_micro_average = f1_score(y_true=v_conc_ground_truth, y_pred=v_conc_pred_one_hot, average='micro', zero_division=0)
        f, pr, rec, fs, prs, recs = F1_as_evaluator(v_conc_pred_one_hot, v_conc_ground_truth, self.target_cols)

        print(f'Epoch: {epoch}, Loss:  {sum(loss_list) / len(loss_list):.2f}, '
              f'Val Loss: {sum(v_loss_list) / len(v_loss_list):.2f}, Val Micro F1: {f1_micro_average:.2f}, '
              f'Val evaluator F1: {f: .2f}')

        return sum(v_loss_list) / len(v_loss_list), f1_micro_average

    def evaluation(self, vloader):

        # Turn off train mode
        self.eval()

        v_loss_list = []
        v_pred_one_hot = []
        v_ground_truth = []
        for _, v_data in tqdm(enumerate(vloader, 0)):
            v_ids = v_data['ids'].to(self.device, dtype=torch.long)
            v_targets = v_data['targets'].to(self.device, dtype=torch.float)

            with torch.no_grad():
                v_outputs = self(v_ids)

                if W_LOSS_WEIGHTS:
                    v_loss_weights_repeated_batch_size = \
                        np.repeat(self.train_dataset.train_loss_weights[np.newaxis, :], v_outputs.size()[0], axis=0)

                    v_final_loss_weights = \
                        np.take_along_axis(v_loss_weights_repeated_batch_size,
                                           np.array(v_targets.cpu().numpy() if v_targets.cpu().numpy().shape[1] == 1
                                                                            else
                                                    np.expand_dims(v_targets.cpu().numpy(), axis=2),
                                                    dtype=int),
                                           axis=1 if v_targets.cpu().numpy().shape[1] == 1 else 2)

                    if v_targets.cpu().numpy().shape[1] > 1:
                        v_final_loss_weights = np.squeeze(v_final_loss_weights, axis=2)

                v_loss = BCE_loss(v_outputs, v_targets,
                                  None if not W_LOSS_WEIGHTS
                                       else torch.from_numpy(v_final_loss_weights).to(self.device, dtype=torch.float32)) \
                        if LOSS == 'BCE' \
                        else (f1_loss(v_outputs, v_targets,
                                      None if not W_LOSS_WEIGHTS
                                           else torch.from_numpy(v_final_loss_weights).to(self.device, dtype=torch.float32))
                              if LOSS == 'sigmoidF1'
                              else 0.0)

                v_loss_list.append(v_loss if LOSS == 'sigmoidF1' else v_loss.item())

                sig = torch.nn.Sigmoid()
                v_probs = sig(v_outputs)

            v_pred_one_hot.append(np.around(v_probs.cpu().numpy()))
            v_ground_truth.append(v_targets.cpu().numpy())

        v_conc_pred_one_hot = np.concatenate(v_pred_one_hot, axis=0)
        v_conc_ground_truth = np.concatenate(v_ground_truth, axis=0)

        return v_conc_pred_one_hot, v_conc_ground_truth, v_loss_list

    def best_model_evaluation(self, vloader, evaluation_only=False, write_file=False):

        if not evaluation_only:
            print('Training has ended. Loading best model for evaluation..')
        else:
            print('Loading saved model for evaluation..')

        self.load_state_dict(torch.load(MODEL_PATH if not evaluation_only else MODEL_PATH_FOR_EVALUATION_ONLY,
                                        map_location=torch.device(self.device)))

        v_conc_pred_one_hot, v_conc_ground_truth, final_v_loss_list = self.evaluation(vloader)

        final_f1_micro_average = f1_score(y_true=v_conc_ground_truth,
                                          y_pred=v_conc_pred_one_hot,
                                          average='micro',
                                          zero_division=0)
        v_clr_dict = classification_report(v_conc_ground_truth,
                                           v_conc_pred_one_hot,
                                           zero_division=0)
        final_f, final_pr, final_rec, final_fs, final_prs, final_recs = \
            F1_as_evaluator(v_conc_pred_one_hot, v_conc_ground_truth, self.target_cols)

        print(f'Best model: \n'
              f'Val Loss: {sum(final_v_loss_list) / len(final_v_loss_list):.2f}, Val micro F1: {final_f1_micro_average:.2f}\n'
              f'Classification Report: \n {v_clr_dict}\n'
              f'Val evaluator F1: {final_f:.2f}, Val evaluator Precision: {final_pr:.2f}, Val evaluator Recall: {final_rec:.2f}\n')

        for value_col_id, value_col in enumerate(self.target_cols):
            print("measure {\n key: \"Precision " + value_col + "\"\n value: \"" + str(final_prs[value_col_id]) + "\"\n}\n" +
                  "measure {\n key: \"Recall " + value_col + "\"\n value: \"" + str(final_recs[value_col_id]) + "\"\n}\n" +
                  "measure {\n key: \"F1 " + value_col + "\"\n value: \"" + str(final_fs[value_col_id]) + "\"\n}\n")

        if write_file:
            writeRun(labels=pd.DataFrame(v_conc_pred_one_hot, columns=self.target_cols),
                     argument_ids=self.loader_valid_dataset.workingTable["Argument ID"].tolist(),
                     outputDataset='./')
            evaluateRun(self.loader_valid_dataset.DATA_PATH + 'validation_labels_only', './', './')

    def train_(self):

        #Create a directory to store the model
        if not os.path.exists('/'.join(MODEL_PATH.split('/')[:-1])):
            os.mkdir('/'.join(MODEL_PATH.split('/')[:-1]))

        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        if self.max_length > 512:
            tokenizer.model_max_length = self.max_length
            print('Max len exceeds 512 tokens !!! Tokenizer is changed !!!')

        self.to(self.device)

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
            val_loss, val_F1 = self.one_epoch_train(epoch, train_loader, valid_loader, optimizer)

            if best_val_loss > val_loss:
                best_val_loss = val_loss
                epochs_wo_improve = 0
                torch.save(self.state_dict(), MODEL_PATH)
            elif epochs_wo_improve > PATIENCE:
                print(f'Early stopping at epoch {epoch} !')
                break
            else:
                epochs_wo_improve += 1

        self.best_model_evaluation(valid_loader)

    def evaluate_(self):

        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        if self.max_length > 512:
            tokenizer.model_max_length = self.max_length
            print('Max len exceeds 512 tokens !!! Tokenizer is changed !!!')

        self.to(self.device)

        if not W_NEW_DATA:
            validation_df_for_BERTDataset = self.dl.validation
        else:
            validation_df_for_BERTDataset = self.loader_valid_dataset.workingTable

        self.valid_dataset = BERTDataset(validation_df_for_BERTDataset, tokenizer,
                                         self.max_length, target_cols=self.target_cols)
        valid_loader = DataLoader(self.valid_dataset, batch_size=TRAIN_BATCH_SIZE,
                                  num_workers=4, shuffle=False, pin_memory=True)

        self.best_model_evaluation(valid_loader, evaluation_only=True, write_file=True)

    def test_(self):

        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        if self.max_length > 512:
            tokenizer.model_max_length = self.max_length
            print('Max len exceeds 512 tokens !!! Tokenizer is changed !!!')

        self.to(self.device)

        test_df_for_BERTDataset = self.loader_test_dataset.workingTable
        self.test_dataset = BERTDataset(test_df_for_BERTDataset, tokenizer,
                                        self.max_length, target_cols=self.target_cols,
                                        test=True)
        test_loader = DataLoader(self.test_dataset, batch_size=TRAIN_BATCH_SIZE,
                                  num_workers=4, shuffle=False, pin_memory=True)

        print('Loading saved model for test..')
        self.load_state_dict(torch.load(MODEL_PATH_FOR_EVALUATION_ONLY,
                                        map_location=torch.device(self.device)))

        # Turn off train mode
        self.eval()

        t_pred_one_hot = []
        for _, t_data in tqdm(enumerate(test_loader, 0)):
            t_ids = t_data['ids'].to(self.device, dtype=torch.long)

            with torch.no_grad():
                t_outputs = self(t_ids)

                sig = torch.nn.Sigmoid()
                t_probs = sig(t_outputs)

            t_pred_one_hot.append(np.around(t_probs.cpu().numpy()))

        t_conc_pred_one_hot = np.concatenate(t_pred_one_hot, axis=0)

        writeRun(labels=pd.DataFrame(t_conc_pred_one_hot, columns=self.target_cols),
                 argument_ids=self.loader_test_dataset.workingTable["Argument ID"].tolist(),
                 outputDataset='./')

if __name__ == '__main__':
    print('BERT Utils')