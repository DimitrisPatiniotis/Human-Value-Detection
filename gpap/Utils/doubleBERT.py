from bert import BERTClass
from dataset_constructor import BERTDataset
from settings import *

import pandas as pd
from transformers import AutoTokenizer, AdamW
from torch.utils.data import DataLoader
import torch
import numpy as np
import os

class DoubleBert(object):

    def __init__(self, target_cols, max_length, device='cpu',
                 loader_train_dataset=None, loader_valid_dataset=None):

        self.loader_train_dataset = loader_train_dataset
        self.loader_valid_dataset = loader_valid_dataset
        self.max_length = max_length
        self.device = device
        self.target_cols = target_cols
        self.train_dataset = None
        self.valid_dataset = None

        # Merge train and validation datasets to train a BERT with the entire dataset
        self.merged_train_df_for_BERTDataset = pd.concat([self.loader_train_dataset.workingTable,
                                                          self.loader_valid_dataset.workingTable],
                                                         axis=0).reset_index(drop=True)

        self.train_df_for_BERTDataset = self.loader_train_dataset.workingTable
        self.validation_df_for_BERTDataset = self.loader_valid_dataset.workingTable

        # Define the BERT models
        self.BERT_for_merged_dataset = BERTClass(self.target_cols, self.max_length, device=self.device,
                                                 loader_train_dataset=None,
                                                 loader_valid_dataset=self.loader_valid_dataset)
        self.BERT_for_original_dataset = BERTClass(self.target_cols, self.max_length, device=self.device,
                                                   loader_train_dataset=self.loader_train_dataset,
                                                   loader_valid_dataset=self.loader_valid_dataset)

        # Define the tokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased' if BERT_MODEL == 'Base' else 'bert-large-uncased')
        if self.max_length > 512:
            tokenizer.model_max_length = self.max_length
            print('Max len exceeds 512 tokens !!! Tokenizer is changed !!!')

        # Define the BERTDataset for each set and for each model
        self.train_dataset_for_BERT_for_merged_dataset = \
            BERTDataset(self.merged_train_df_for_BERTDataset.copy(), tokenizer, self.max_length,
                        target_cols=self.target_cols, train=True)
        self.valid_dataset_for_BERT_for_merged_dataset = \
            BERTDataset(self.validation_df_for_BERTDataset.copy(), tokenizer, self.max_length,
                        target_cols=self.target_cols)
        self.BERT_for_merged_dataset.train_dataset = self.train_dataset_for_BERT_for_merged_dataset

        self.train_dataset_for_BERT_for_original_dataset = \
            BERTDataset(self.train_df_for_BERTDataset.copy(), tokenizer, self.max_length,
                        target_cols=self.target_cols, train=True)
        self.valid_dataset_for_BERT_for_original_dataset = \
            BERTDataset(self.validation_df_for_BERTDataset.copy(), tokenizer, self.max_length,
                        target_cols=self.target_cols)

    def train_(self):

        # Create a directory to store the model
        if not os.path.exists('/'.join(MODEL_PATH.split('/')[:-1])):
            os.mkdir('/'.join(MODEL_PATH.split('/')[:-1]))

        # Define the train loaders
        train_loader_for_BERT_for_merged_dataset = \
            DataLoader(self.train_dataset_for_BERT_for_merged_dataset,
                       batch_size=TRAIN_BATCH_SIZE,
                       num_workers=4, shuffle=True, pin_memory=True)
        valid_loader_for_BERT_for_merged_dataset = \
            DataLoader(self.valid_dataset_for_BERT_for_merged_dataset,
                       batch_size=TRAIN_BATCH_SIZE,
                       num_workers=4, shuffle=False, pin_memory=True)
        train_loader_for_BERT_for_original_dataset = \
            DataLoader(self.train_dataset_for_BERT_for_original_dataset,
                       batch_size=TRAIN_BATCH_SIZE,
                       num_workers=4, shuffle=True, pin_memory=True)
        valid_loader_for_BERT_for_original_dataset = \
            DataLoader(self.valid_dataset_for_BERT_for_original_dataset,
                       batch_size=TRAIN_BATCH_SIZE,
                       num_workers=4, shuffle=False, pin_memory=True)

        # Load the models to GPU if available
        self.BERT_for_merged_dataset.to(self.device)
        self.BERT_for_original_dataset.to(self.device)

        # Define the optimizers
        optimizer_for_BERT_for_merged_dataset = \
            AdamW(params=self.BERT_for_merged_dataset.parameters(), lr=LEARNING_RATE,
                  weight_decay=1e-6, no_deprecation_warning=True)
        optimizer_for_BERT_for_original_dataset = \
            AdamW(params=self.BERT_for_original_dataset.parameters(), lr=LEARNING_RATE,
                  weight_decay=1e-6, no_deprecation_warning=True)

        print('Start training...')
        base_path = MODEL_PATH[0:-len(MODEL_PATH.split('/')[-1])]
        best_val_loss = np.inf
        epochs_wo_improve = 0
        for epoch in range(EPOCHS):

            print("\nTurn of BERT with merged dataset:")
            val_loss_for_BERT_with_merged_dataset, val_F1_for_BERT_with_merged_dataset = \
                self.BERT_for_merged_dataset.one_epoch_train(epoch,
                                                             train_loader_for_BERT_for_merged_dataset,
                                                             valid_loader_for_BERT_for_merged_dataset,
                                                             optimizer_for_BERT_for_merged_dataset)

            print("\nTurn of BERT with original dataset:")
            val_loss_for_BERT_with_original_dataset, val_F1_for_BERT_with_original_dataset = \
                self.BERT_for_original_dataset.one_epoch_train(epoch,
                                                             train_loader_for_BERT_for_original_dataset,
                                                             valid_loader_for_BERT_for_original_dataset,
                                                             optimizer_for_BERT_for_original_dataset)

            if best_val_loss > val_loss_for_BERT_with_original_dataset:
                best_val_loss = val_loss_for_BERT_with_original_dataset
                epochs_wo_improve = 0
                torch.save(self.BERT_for_merged_dataset.state_dict(), base_path + 'BERT_for_merged_dataset.pt')
                torch.save(self.BERT_for_original_dataset.state_dict(), base_path + 'BERT_for_original_dataset.pt')
            elif epochs_wo_improve > PATIENCE:
                print(f'Early stopping at epoch {epoch} !')
                break
            else:
                epochs_wo_improve += 1

        print('Best model evaluation...')
        self.BERT_for_merged_dataset.best_model_evaluation(valid_loader_for_BERT_for_merged_dataset,
                                                           path=base_path + 'BERT_for_merged_dataset.pt')
        self.BERT_for_original_dataset.best_model_evaluation(valid_loader_for_BERT_for_original_dataset,
                                                             path=base_path + 'BERT_for_original_dataset.pt')