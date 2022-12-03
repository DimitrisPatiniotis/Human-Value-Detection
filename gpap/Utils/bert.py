import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AdamW
from transformers import BertModel

from settings import *


def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

class BERTDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, target_cols):
        self.df = df
        self.max_len = max_len
        self.text = df.Text
        self.tokenizer = tokenizer
        # Bug Here
        
        self.targets = df[target_cols].values
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        text = self.text[index]
        inputs = self.tokenizer.encode_plus(text,
                                            truncation=True,
                                            add_special_tokens=False,
                                            max_length=self.max_len,
                                            padding='max_length',
                                            return_token_type_ids=False)
        ids = inputs['input_ids']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

class BERTClass(torch.nn.Module):
    def __init__(self, dl, target_cols, max_length, device='cpu'):

        super(BERTClass, self).__init__()

        self.dl = dl
        self.max_length = max_length
        self.device = device
        self.freeze_bert = FREEZE_BERT
        self.head_type = HEAD_TYPE
        self.multihead = MULTIHEAD
        self.biodirectional_GRU = BIODIRECTIONAL_GRU
        self.dropout_rate = DROPOUT
        self.GRU_hidden_dim = GRU_HIDDEN_DIM

        self.bert = BertModel.from_pretrained('bert-base-uncased')

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
                self.fc = torch.nn.Linear(embedding_dim*max_length, len(target_cols))
            else:
                self.fcs = [torch.nn.Linear(embedding_dim*max_length, 1, device=self.device)
                            for _ in range(len(target_cols))]
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
                self.rnns = [torch.nn.GRU(embedding_dim,
                                          self.GRU_hidden_dim,
                                          num_layers=1,
                                          bidirectional=self.biodirectional_GRU,
                                          batch_first=True,
                                          dropout=0,
                                          device=self.device) for _ in range(len(target_cols))]
                self.fcs = [torch.nn.Linear(self.GRU_hidden_dim*2 if self.biodirectional_GRU
                                                                  else self.GRU_hidden_dim,
                                            1,
                                            device=self.device)
                            for _ in range(len(target_cols))]

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
            flat_embedded = torch.flatten(embedded, start_dim=1)
            # flat_embedded = [batch size, sent len * emb dim]

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

            loss = loss_fn(outputs, targets)
            loss_list.append(loss.item())

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

                v_loss = loss_fn(v_outputs, v_targets)
                v_loss_list.append(v_loss.item())

                sig = torch.nn.Sigmoid()
                v_probs = sig(v_outputs)

            v_pred_one_hot.append(np.around(v_probs.cpu().numpy()))
            v_ground_truth.append(v_targets.cpu().numpy())

        v_conc_pred_one_hot = np.concatenate(v_pred_one_hot, axis=0)
        v_conc_ground_truth = np.concatenate(v_ground_truth, axis=0)
        v_clr_dict = classification_report(v_conc_ground_truth, v_conc_pred_one_hot, output_dict=True)

        print(f'Epoch: {epoch}, Loss:  {sum(loss_list) / len(loss_list):.2f}, '
              f'Val Loss: {sum(v_loss_list) / len(v_loss_list):.2f}, Val F1: {v_clr_dict["macro avg"]["f1-score"]:.2f}')

        return sum(v_loss_list) / len(v_loss_list), v_clr_dict["macro avg"]["f1-score"]


    def train_(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        if self.max_length > 512:
            tokenizer.model_max_length = self.max_length
            print('Max len exceeds 512 tokens !!! Tokenizer is changed !!!')

        self.to(device)

        optimizer = AdamW(params=self.parameters(), lr=LEARNING_RATE, weight_decay=1e-6, no_deprecation_warning=True)

        train_dataset = BERTDataset(self.dl.train, tokenizer, self.max_length, target_cols=self.dl.get_target_cols())
        valid_dataset = BERTDataset(self.dl.validation, tokenizer, self.max_length, target_cols=self.dl.get_target_cols())

        train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, num_workers=4, shuffle=True,
                                  pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=TRAIN_BATCH_SIZE, num_workers=4, shuffle=False,
                                  pin_memory=True)

        print('Starting training...')
        best_val_loss = np.inf
        epochs_wo_improve = 0
        for epoch in range(EPOCHS):
            val_loss, val_F1 = self.one_epoch_train(epoch, train_loader, valid_loader, device, optimizer)

            if best_val_loss > val_loss:
                epochs_wo_improve = 0
                torch.save(self.state_dict(), './../saved_model/model.pt')
            elif epochs_wo_improve > PATIENCE:
                print(f'Early stopping at epoch {epoch} !')
                break
            else:
                epochs_wo_improve += 1

    
if __name__ == '__main__':
    print('BERT Utils')