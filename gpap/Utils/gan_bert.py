"""
Code reference: https://github.com/crux82/ganbert-pytorch/blob/main/GANBERT_pytorch.ipynb
Paper reference:

@inproceedings{croce-etal-2020-gan,
    title = "{GAN}-{BERT}: Generative Adversarial Learning for Robust Text Classification with a Bunch of Labeled Examples",
    author = "Croce, Danilo  and
      Castellucci, Giuseppe  and
      Basili, Roberto",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.191",
    pages = "2114--2119"
}

MODIFICATIONS HAVE BEEN APPLIED.
"""

import torch
from transformers import AutoTokenizer, AdamW
from bert import *
from dataset_constructor import BERTDataset

class Generator(torch.nn.Module):
    def __init__(self, noise_size=100, output_size=768, hidden_size=512, dropout_rate=GEN_DROPOUT_RATE):
        super(Generator, self).__init__()

        self.first_linear = torch.nn.Linear(noise_size, hidden_size)
        self.first_LeakyReLU = torch.nn.LeakyReLU(0.2, inplace=True)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.second_linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, noise):

        first_linear_out = self.first_linear(noise)
        first_LeakyReLU_out = self.first_LeakyReLU(first_linear_out)
        dropout_out = self.dropout(first_LeakyReLU_out)
        output_rep = self.second_linear(dropout_out)

        return output_rep

class Discriminator(torch.nn.Module):
    def __init__(self, input_size=768, hidden_size=512, num_labels=1, dropout_rate=DISCR_DROPOUT_RATE):
        super(Discriminator, self).__init__()

        self.input_dropout = torch.nn.Dropout(p=dropout_rate)
        self.first_linear = torch.nn.Linear(input_size, hidden_size)
        self.first_LeakyReLU = torch.nn.LeakyReLU(0.2, inplace=True)
        self.first_dropout = torch.nn.Dropout(dropout_rate)
        self.logit = torch.nn.Linear(hidden_size, num_labels+1) # +1 for the probability of this sample being fake/real.

    def forward(self, input_rep):

        input_dropout_out = self.input_dropout(input_rep)
        first_linear_out = self.first_linear(input_dropout_out)
        first_LeakyReLU_out = self.first_LeakyReLU(first_linear_out)
        last_rep = self.first_dropout(first_LeakyReLU_out)
        logits = self.logit(last_rep)

        return last_rep, logits

class GanBERT(object):
    def __init__(self, target_cols=None, max_length=None, dl=None, loader_train_dataset=None,
                 loader_valid_dataset=None, device='cpu',
                 discr_hidden_size=512, num_labels=1, gen_noise_size=100, gen_hidden_size=512,
                 epsilon=1e-8):

        self.target_cols = target_cols
        self.max_length = max_length
        self.dl = dl
        self.loader_train_dataset = loader_train_dataset
        self.loader_valid_dataset = loader_valid_dataset
        self.device = device
        self.discr_lr = DISCR_LR
        self.gen_lr = GEN_LR
        self.gen_noise_size = gen_noise_size
        self.epsilon = epsilon
        self.gen_optimizer = None
        self.discr_optimizer = None
        self.train_dataset = None
        self.valid_dataset = None

        # Define the base BERT model (to get its last layer hidden state of CLS), Generator, and Discriminator
        self.bert_model_embeddings = BERTClass(self.target_cols, self.max_length, self.dl, self.loader_train_dataset,
                                               self.loader_valid_dataset, self.device)
        discr_input_size = gen_output_size = self.bert_model_embeddings.embedding_dim

        self.generator = Generator(noise_size=gen_noise_size, output_size=gen_output_size, hidden_size=gen_hidden_size)

        self.discriminator = Discriminator(input_size=discr_input_size, hidden_size=discr_hidden_size,
                                           num_labels=num_labels)

        self.sig = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=-1)

    def train_(self):

        #Create a directory to store the model
        if not os.path.exists('/'.join(MODEL_PATH.split('/')[:-1])):
            os.mkdir('/'.join(MODEL_PATH.split('/')[:-1]))

        #Define the tokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        if self.max_length > 512:
            tokenizer.model_max_length = self.max_length
            print('Max len exceeds 512 tokens !!! Tokenizer is changed !!!')

        # Load all model to GPU (if available)
        self.bert_model_embeddings.to(self.device)
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        # Get the parameters of each model
        bert_model_embeddings_vars = [i for i in self.bert_model_embeddings.parameters()]
        discr_vars = bert_model_embeddings_vars + [v for v in self.discriminator.parameters()]
        gen_vars = [v for v in self.generator.parameters()]

        # Define the optimizer of each model
        self.discr_optimizer = torch.optim.AdamW(discr_vars, lr=self.discr_lr)
        self.gen_optimizer = torch.optim.AdamW(gen_vars, lr=self.gen_lr)

        if not W_NEW_DATA:
            train_df_for_BERTDataset = self.dl.train
            validation_df_for_BERTDataset = self.dl.validation
        else:
            train_df_for_BERTDataset = self.loader_train_dataset.workingTable
            validation_df_for_BERTDataset = self.loader_valid_dataset.workingTable

        self.train_dataset = BERTDataset(train_df_for_BERTDataset, tokenizer, self.max_length,
                                         target_cols=self.target_cols, train=True,
                                         unlabeled_df=None if not W_UNLABELED_DATA
                                                           else self.loader_train_dataset.workingTable_Unlabeled)
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
            val_loss, val_F1 = self.one_epoch_train(epoch, train_loader, valid_loader)

            if best_val_loss > val_loss:
                best_val_loss = val_loss
                epochs_wo_improve = 0

                # Save models
                base_path = MODEL_PATH[0:-len(MODEL_PATH.split('/')[-1])]

                torch.save(self.bert_model_embeddings.state_dict(),
                           base_path + 'bert_embeddings_model.pt')
                torch.save(self.generator.state_dict(),
                           base_path + 'generator_model.pt')
                torch.save(self.discriminator.state_dict(),
                           base_path + 'discriminator_model.pt')

            elif epochs_wo_improve > PATIENCE:
                print(f'Early stopping at epoch {epoch} !')
                break

            else:
                epochs_wo_improve += 1

        self.best_model_evaluation(valid_loader)

    def one_epoch_train(self, epoch, tloader, vloader):

        ######## Train the model ########

        #Set all models to training mode
        self.bert_model_embeddings.train()
        self.generator.train()
        self.discriminator.train()

        D_L_Supervised_list = []
        D_L_unsupervised1U_list = []
        D_L_unsupervised2U_list = []
        d_loss_list = []
        g_loss_d_list = []
        g_feat_reg_list = []
        g_loss_list = []
        for _, data in tqdm(enumerate(tloader, 0)):
            ids = data['ids'].to(self.device, dtype=torch.long)
            targets = data['targets'].to(self.device, dtype=torch.float)
            if W_UNLABELED_DATA:
                label_mask = data['label_mask'].to(self.device, dtype=torch.bool)

            real_batch_size = targets.shape[0]

            # Get the BERT hidden state for the labeled real data
            bert_outputs = self.bert_model_embeddings(ids)

            # Create noise to be used as input for the Generator
            noise = torch.zeros(real_batch_size, self.gen_noise_size, device=self.device).uniform_(0, 1)

            # Generate Fake data
            gen_rep = self.generator(noise)

            # Get the output of the Discriminator for real and fake data.
            # First, we put together the output of BERT and Generator
            discriminator_input = torch.cat([bert_outputs, gen_rep], dim=0)
            # Then, we select the output of Discriminator
            features, logits = self.discriminator(discriminator_input)

            # Finally, we separate the discriminator's output for the real and fake data
            features_list = torch.split(features, real_batch_size)
            D_real_features = features_list[0]
            D_fake_features = features_list[1]

            logits_list = torch.split(logits, real_batch_size)
            D_real_logits = logits_list[0]
            D_fake_logits = logits_list[1]
            if W_UNLABELED_DATA:
                D_real_logits_for_D_L_Supervised = D_real_logits[label_mask]
                targets_for_D_L_Supervised = targets[label_mask]
            else:
                D_real_logits_for_D_L_Supervised = D_real_logits
                targets_for_D_L_Supervised = targets

            D_real_probs = self.sig(D_real_logits) if FAKE_OR_REAL_PROBS_METHOD == 'BCE' else self.softmax(D_real_logits)
            D_fake_probs = self.sig(D_fake_logits) if FAKE_OR_REAL_PROBS_METHOD == 'BCE' else self.softmax(D_fake_logits)

            # ---------------------------------
            #  LOSS evaluation
            # ---------------------------------

            # Calculate the corresponding loss (BCE or sigmoidF1) for the multilabel task
            # (and the corresponding weights if W_LOSS_WEIGHTS).
            # This loss is the supervised term of Discriminator's loss
            if W_LOSS_WEIGHTS:
                loss_weights_repeated_batch_size = \
                    np.repeat(self.train_dataset.train_loss_weights[np.newaxis, :],
                              D_real_logits_for_D_L_Supervised[:, 0:-1].size()[0],
                              axis=0)

                final_loss_weights = \
                    np.take_along_axis(loss_weights_repeated_batch_size,
                                       np.array(targets_for_D_L_Supervised.cpu().numpy()
                                                if targets_for_D_L_Supervised.cpu().numpy().shape[1] == 1
                                                else
                                                np.expand_dims(targets_for_D_L_Supervised.cpu().numpy(), axis=2),
                                                dtype=int),
                                       axis=1 if targets_for_D_L_Supervised.cpu().numpy().shape[1] == 1 else 2)

                if targets_for_D_L_Supervised.cpu().numpy().shape[1] > 1:
                    final_loss_weights = np.squeeze(final_loss_weights, axis=2)

            D_L_Supervised = BCE_loss(D_real_logits_for_D_L_Supervised[:, 0:-1], targets_for_D_L_Supervised,
                                      None if not W_LOSS_WEIGHTS
                                           else torch.from_numpy(final_loss_weights).to(self.device, dtype=torch.float32)) \
                             if LOSS == 'BCE' \
                             else (f1_loss(D_real_logits_for_D_L_Supervised[:, 0:-1], targets_for_D_L_Supervised,
                                           None if not W_LOSS_WEIGHTS
                                                else torch.from_numpy(final_loss_weights).to(self.device, dtype=torch.float32))
                                    if LOSS == 'sigmoidF1'
                                    else 0.0)

            D_L_unsupervised1U = -1 * torch.mean(torch.log(1 - D_real_probs[:, -1] + self.epsilon))
            D_L_unsupervised2U = -1 * torch.mean(torch.log(D_fake_probs[:, -1] + self.epsilon))
            d_loss = (D_L_Supervised if str(D_L_Supervised.item()) != 'nan' else 0) + D_L_unsupervised1U + D_L_unsupervised2U

            # Store discriminator's loss as well as all its terms separately
            if str(D_L_Supervised.item()) != 'nan':
                D_L_Supervised_list.append(D_L_Supervised.item())
            D_L_unsupervised1U_list.append(D_L_unsupervised1U.item())
            D_L_unsupervised2U_list.append(D_L_unsupervised2U.item())
            d_loss_list.append(d_loss.item())

            # Generator's LOSS estimation
            g_loss_d = -1 * torch.mean(torch.log(1 - D_fake_probs[:, -1] + self.epsilon))
            g_feat_reg = \
                torch.mean(torch.pow(torch.mean(D_real_features, dim=0) - torch.mean(D_fake_features, dim=0), 2))
            g_loss = g_loss_d + g_feat_reg

            # Store generator's loss as well as all its terms separately
            g_loss_d_list.append(g_loss_d.item())
            g_feat_reg_list.append(g_feat_reg.item())
            g_loss_list.append(g_loss.item())

            # ---------------------------------
            #  OPTIMIZATION
            # ---------------------------------
            # Avoid gradient accumulation
            self.gen_optimizer.zero_grad()
            self.discr_optimizer.zero_grad()

            # Calculate weight updates
            # retain_graph=True is required since the underlying graph will be deleted after backward
            g_loss.backward(retain_graph=True)
            d_loss.backward()

            # Apply modifications
            self.gen_optimizer.step()
            self.discr_optimizer.step()

        ######## Evaluate the model ########
        print('Evaluating...')

        v_conc_pred_one_hot, v_conc_ground_truth, v_loss_list = self.evaluation(vloader)

        f1_micro_average = f1_score(y_true=v_conc_ground_truth, y_pred=v_conc_pred_one_hot, average='micro', zero_division=0)
        f, pr, rec, fs, prs, recs = F1_as_evaluator(v_conc_pred_one_hot, v_conc_ground_truth, self.target_cols)

        print(f'Epoch: {epoch}\n '
              f'Discriminator Supervised Loss:  {sum(D_L_Supervised_list) / len(D_L_Supervised_list):.2f}\n '
              f'Discriminator Unsupervised Real Data Loss: {sum(D_L_unsupervised1U_list) / len(D_L_unsupervised1U_list):.2f}\n '
              f'Discriminator Unsupervised Fake Data Loss: {sum(D_L_unsupervised2U_list) / len(D_L_unsupervised2U_list):.2f}\n '
              f'Total Discriminator Loss: {sum(d_loss_list) / len(d_loss_list):.2}\n '
              f'Generator Fake Data Loss: {sum(g_loss_d_list) / len(g_loss_d_list):.2f}\n '
              f'Generator Features Regularization Loss: {sum(g_feat_reg_list) / len(g_feat_reg_list):.2f}\n '
              f'Total Generator Loss: {sum(g_loss_list) / len(g_loss_list)} \n'
              f'Val Loss: {sum(v_loss_list) / len(v_loss_list):.2f}, '
              f'Val Micro F1: {f1_micro_average:.2f}, '
              f'Val evaluator F1: {f: .2f}')

        return sum(v_loss_list) / len(v_loss_list), f1_micro_average

    def evaluation(self, vloader):

        # Set all models to evaluation mode
        self.bert_model_embeddings.eval()
        self.generator.eval()
        self.discriminator.eval()

        v_loss_list = []
        v_pred_one_hot = []
        v_ground_truth = []
        for _, v_data in tqdm(enumerate(vloader, 0)):
            v_ids = v_data['ids'].to(self.device, dtype=torch.long)
            v_targets = v_data['targets'].to(self.device, dtype=torch.float)

            with torch.no_grad():
                # Get the BERT hidden state for the labeled real data of validation set
                v_bert_outputs = self.bert_model_embeddings(v_ids)

                # Get the output of the Discriminator only for real data of validation set.
                _, v_logits = self.discriminator(v_bert_outputs)
                # Filter Discriminator's output (keep only the value classes)
                v_outputs = v_logits[:, 0:-1]

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

    def best_model_evaluation(self, vloader, evaluation_only=False):

        if not evaluation_only:
            print('Training has ended. Loading best model for evaluation..')
        else:
            print('Loading saved model for evaluation..')

        # Load trained models of BERT embeddings and Discriminator
        base_path = MODEL_PATH[0:-len(MODEL_PATH.split('/')[-1])] \
                    if not evaluation_only \
                    else \
                    MODEL_PATH_FOR_EVALUATION_ONLY[0:-len(MODEL_PATH_FOR_EVALUATION_ONLY.split('/')[-1])]

        self.bert_model_embeddings.load_state_dict(torch.load(base_path + 'bert_embeddings_model.pt',
                                                              map_location=torch.device(self.device)))
        self.discriminator.load_state_dict(torch.load(base_path + 'discriminator_model.pt',
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
              f'Val Loss: {sum(final_v_loss_list) / len(final_v_loss_list):.2f}, '
              f'Val micro F1: {final_f1_micro_average:.2f}\n'
              f'Classification Report: \n {v_clr_dict}\n'
              f'Val evaluator F1: {final_f:.2f}, '
              f'Val evaluator Precision: {final_pr:.2f}, '
              f'Val evaluator Recall: {final_rec:.2f}\n')

        for value_col_id, value_col in enumerate(self.target_cols):
            print("measure {\n key: \"Precision " + value_col + "\"\n value: \"" + str(final_prs[value_col_id]) + "\"\n}\n" +
                  "measure {\n key: \"Recall " + value_col + "\"\n value: \"" + str(final_recs[value_col_id]) + "\"\n}\n" +
                  "measure {\n key: \"F1 " + value_col + "\"\n value: \"" + str(final_fs[value_col_id]) + "\"\n}\n")

    def evaluate_(self):

        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        if self.max_length > 512:
            tokenizer.model_max_length = self.max_length
            print('Max len exceeds 512 tokens !!! Tokenizer is changed !!!')

        # Load all model to GPU (if available)
        self.bert_model_embeddings.to(self.device)
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        if not W_NEW_DATA:
            validation_df_for_BERTDataset = self.dl.validation
        else:
            validation_df_for_BERTDataset = self.loader_valid_dataset.workingTable

        self.valid_dataset = BERTDataset(validation_df_for_BERTDataset, tokenizer, self.max_length,
                                         target_cols=self.target_cols)
        valid_loader = DataLoader(self.valid_dataset, batch_size=TRAIN_BATCH_SIZE,
                                  num_workers=4, shuffle=False, pin_memory=True)

        self.best_model_evaluation(valid_loader, evaluation_only=True)
