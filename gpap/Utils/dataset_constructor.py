import pandas as pd

from settings import *

from torch.utils.data import Dataset
import numpy as np
import torch

class BERTDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, target_cols, train=False, test=False, unlabeled_df=None):
        self.df = df
        self.max_len = max_len
        self.text = df.Text
        self.tokenizer = tokenizer
        self.targets = None if test else df[target_cols].values
        self.train = train
        self.test = test
        self.unlabeled_text = \
            None if not self.train or not W_UNLABELED_DATA\
                 else pd.concat([unlabeled_df[:]]*UPSAMPLING_RATE_UNLABELED_DATA, ignore_index=True).Text

        if self.train and W_LOSS_WEIGHTS:

            if not SINGLE_CLASS:
                num_train_samples_per_binary_class_per_value_class = df[target_cols].\
                    apply(lambda x: x.value_counts()).values

                train_loss_weights_positive_class_per_value_class = \
                    (1 - (num_train_samples_per_binary_class_per_value_class[1] /
                          num_train_samples_per_binary_class_per_value_class.sum(axis=0)))

                train_loss_weights_negative_class_per_value_class = \
                    (1 - (num_train_samples_per_binary_class_per_value_class[0] /
                          num_train_samples_per_binary_class_per_value_class.sum(axis=0)))

                self.train_loss_weights = np.concatenate(
                    (np.array([train_loss_weights_negative_class_per_value_class]).T,
                     np.array([train_loss_weights_positive_class_per_value_class]).T), axis=1)

            else:
                # Calculations based on:
                # https://naadispeaks.wordpress.com/2021/07/31/handling-imbalanced-classes-with-weighted-loss-in-pytorch/

                num_train_samples_per_binary_class = df[target_cols].value_counts()
                self.train_loss_weights = \
                    (1 - (num_train_samples_per_binary_class / num_train_samples_per_binary_class.sum())).values

        if self.train and W_UNLABELED_DATA:

            self.label_mask = np.concatenate([np.ones(self.targets.shape[0], dtype=bool),
                                              np.zeros(len(self.unlabeled_text), dtype=bool)])
            self.targets = np.concatenate([self.targets,
                                           np.array([[np.nan]*len(target_cols)]*len(self.unlabeled_text))],
                                          axis=0)
            self.text = pd.concat([self.text, self.unlabeled_text], axis=0).reset_index(drop=True)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = self.text[index]

        # Replace some words with '[MASK]'
        if self.train and W_MASKING:

            # Create an array of indices of the worlds to be masked and mask the 'text'
            np_of_words = np.array(list(text.split(' ')))
            masked_np_of_words = np_of_words.copy()
            mask_indices = np.random.randint(np_of_words.shape[0], size=(round(MAKING_PERC * np_of_words.shape[0])))
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
            'targets': [] if self.test else torch.tensor(self.targets[index], dtype=torch.float),
            'label_mask': [] if not W_UNLABELED_DATA or not self.train
                             else torch.tensor(self.label_mask[index], dtype=torch.bool)
                }