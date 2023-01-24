import json
import re
from time import time

import pandas as pd
from matplotlib import pyplot as plt
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, StratifiedKFold
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import seaborn as sns

from settings import *

def get_main_label_names(json_path):
    with open(json_path) as jsonFile:
        data = json.load(jsonFile)
    return list(data.keys())

def load_data(file_path, sep='\t', print_info=True):
    if print_info:
        print('Loading data...')
        starting_time = time()
    content = pd.read_csv(file_path, sep=sep, on_bad_lines='skip')
    if print_info:
        ending_time = time()
        print('Data loaded in {} seconds.'.format(round(ending_time-starting_time, 4)))
    return content

def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text

def set_stance(stance):
    st = 0 if stance == 'against' else 1
    return st

stemmer = PorterStemmer()
def analytical_stem(txt):
    return stemmer.stem(txt)

class Loader():

    def __init__(self,
                 data_base_path='../Data/',
                 ta_file_name='arguments-training.tsv',
                 tl_file_name='labels-training.tsv',
                 tl_lvl1_file_name='level1-labels-training.tsv',
                 workingTableType='value_categories',
                 unlabeled_data_file_name='arguments-test.tsv',
                 with_unlabeled_data=False):

        self.TA_FILE_NAME = ta_file_name
        self.TL_FILE_NAME = tl_file_name
        self.TL_LVL1_FILE_NAME = tl_lvl1_file_name
        self.UNLABELED_DATA_FILE_NAME = unlabeled_data_file_name
        self.WITH_UNLABELED_DATA = with_unlabeled_data
        self.DATA_PATH = data_base_path
        self.workingTableType = workingTableType
        self.stemmer = PorterStemmer()
        self.label_names = None
        self.arguments = None
        self.labels = None
        self.l1labels = None
        self.workingTable = None
        self.train = None
        self.validation = None
        self.target_cols = None
        self.workingTable_Unlabeled = None

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_val = None
        self.y_val = None

    def load(self, clean=True, w_sep=False, w_concl=True, w_stance=True):
        self.arguments = load_data(self.DATA_PATH + self.TA_FILE_NAME)
        if self.workingTableType == 'value_categories' and self.TL_FILE_NAME is not None:
            self.labels = load_data(self.DATA_PATH + self.TL_FILE_NAME)
            # Merging
            self.workingTable = pd.merge(self.arguments, self.labels, on='Argument ID')
            self.label_names = get_main_label_names('../Data/value-categories.json')
        elif self.workingTableType == 'value_categories' and self.TL_FILE_NAME is None:
            self.workingTable = self.arguments
            self.label_names = get_main_label_names('../Data/value-categories.json')
        elif self.workingTableType == '':
            self.labels = load_data(self.DATA_PATH + self.TL_LVL1_FILE_NAME)
            self.workingTable = pd.merge(self.arguments, self.labels, on='Argument ID')

        # Add padding or [SEP]
        self.workingTable['Conclusion'] = self.workingTable['Conclusion'].\
            apply(lambda txt: (txt + (' [SEP] ' if w_sep else ' ')) if not w_stance else txt)
        self.workingTable['Stance'] = self.workingTable['Stance'].\
            apply(lambda txt: ((' [SEP] ' if w_sep else ' ') + txt + ' ')
                              if w_concl else (txt + ' '))

        self.workingTable['Text'] = (self.workingTable['Conclusion'] if w_concl else '') + \
                                    (self.workingTable['Stance'] if w_stance else '') + \
                                    self.workingTable['Premise']
        self.workingTable = self.workingTable.drop(['Conclusion', 'Stance', 'Premise'], axis=1)

        if clean:
            self.workingTable['Text'] = self.workingTable['Text'].apply(clean_text)

        if self.WITH_UNLABELED_DATA:

            self.workingTable_Unlabeled = load_data(self.DATA_PATH + self.UNLABELED_DATA_FILE_NAME)

            # Add padding or [SEP]
            self.workingTable_Unlabeled['Conclusion'] = self.workingTable_Unlabeled['Conclusion']. \
                apply(lambda txt: (txt + (' [SEP] ' if w_sep else ' ')) if not w_stance else txt)
            self.workingTable_Unlabeled['Stance'] = self.workingTable_Unlabeled['Stance']. \
                apply(lambda txt: ((' [SEP] ' if w_sep else ' ') + txt + ' ')
                                  if w_concl else (txt + ' '))

            self.workingTable_Unlabeled['Text'] = (self.workingTable_Unlabeled['Conclusion'] if w_concl else '') + \
                                                  (self.workingTable_Unlabeled['Stance'] if w_stance else '') + \
                                                  self.workingTable_Unlabeled['Premise']
            self.workingTable_Unlabeled = self.workingTable_Unlabeled.drop(['Conclusion', 'Stance', 'Premise'], axis=1)

            if clean:
                self.workingTable_Unlabeled['Text'] = self.workingTable_Unlabeled['Text'].apply(clean_text)

    def get_max_len(self):
        # Find the sentence with the maximum words
        max_words = 0
        idx_max_words = -1
        for idx_row, row in enumerate(self.workingTable['Text']):
            if len(row) > max_words:
                max_words = len(row)
                idx_max_words = idx_row
        return max_words, self.workingTable['Text'].iloc[idx_max_words]
    
    def tknz(self):
        self.workingTable['Text'] = self.workingTable['Text'].apply(print, word_tokenize)
    
    def stem(self):
        self.workingTable['Text'] = self.workingTable['Text'].apply(analytical_stem)
    
    def train_test_validate_split(self, test_size=0.3, to_loader=False, random_state=1):
        X, y = self.workingTable['Text'], self.workingTable[self.label_names]

        if not to_loader:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size,
                                                                                    random_state=random_state, shuffle=True)

        else:

            self.x_train, self.y_train, self.x_test, self.y_test = iterative_train_test_split(self.workingTable[['Text']].values,
                                                                                              self.workingTable[self.label_names].values,
                                                                                              test_size=test_size)

            self.train = pd.concat([pd.DataFrame(self.x_train, columns=['Text']), pd.DataFrame(self.y_train, columns=self.label_names)],
                                   axis=1).reset_index(drop=True)
            self.validation = pd.concat([pd.DataFrame(self.x_test, columns=['Text']), pd.DataFrame(self.y_test, columns=self.label_names)],
                                        axis=1).reset_index(drop=True)

    def multilabel_to_string(self, y):
        return '-'.join(str(int(l.item())) for l in y)

    def multilabel_to_multiclass(self, y):
        y_new = LabelEncoder().fit_transform([self.multilabel_to_string(l) for l in y])
        return y_new

    def StratifiedCFV(self, n_folds=1, random_state=2022):

        if n_folds < 2:
            X_train, X_val, y_train, y_val = \
                train_test_split(np.arange(len(self.workingTable['Text'].index)),
                                 np.zeros(len(self.workingTable['Text'].index)),
                                 test_size=0.1, random_state=random_state, shuffle=True)

            self.train = pd.concat([pd.DataFrame(self.workingTable['Text'].iloc[X_train], columns=['Text']),
                                    pd.DataFrame(self.workingTable[self.label_names].iloc[X_train], columns=self.label_names)],
                                   axis=1).reset_index(drop=True)

            self.validation = pd.concat([pd.DataFrame(self.workingTable['Text'].iloc[X_val], columns=['Text']),
                                         pd.DataFrame(self.workingTable[self.label_names].iloc[X_val], columns=self.label_names)],
                                        axis=1).reset_index(drop=True)

        else:

            folds = StratifiedKFold(n_splits=n_folds, random_state=random_state)

            splits = folds.split(np.zeros(len(self.workingTable['Text'].index)),
                                 self.multilabel_to_multiclass(self.workingTable[self.label_names].values))

            return splits

    def split_to_train_val(self):
        self.workingTable = self.workingTable.sample(frac=1).reset_index(drop=True)
        self.validation = self.workingTable[1:round(len(self.workingTable) * 0.3)].reset_index(drop=True)
        self.train = self.workingTable[round(len(self.workingTable) * 0.3):].reset_index(drop=True)
    
    def show_label_stats(self):
        label_names = get_main_label_names('../Data/value-categories.json')
        total_count = []
        for label in label_names:
            total_count.append(self.workingTable[label].value_counts()[1])        
        fig, ax = plt.subplots(figsize=(16, 9))
        # Horizontal Bar Plot
        ax.barh(label_names, total_count)
        for s in ['top', 'bottom', 'left', 'right']:
            ax.spines[s].set_visible(False)
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_tick_params(pad=5)
        ax.yaxis.set_tick_params(pad=10)
        ax.grid(b=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.2)
        ax.invert_yaxis()
        for i in ax.patches:
            plt.text(i.get_width()+0.2, i.get_y()+0.5, str(round((i.get_width()), 2)), fontsize=10, fontweight='bold', color='grey')
        ax.set_title('Occurrences of Value Categories', loc='left')
        plt.show()

    def plot_sentences_length(self):
        fig, ax = plt.subplots(figsize=(15, 15))
        ax.set_title('Histogram of sentences length', loc='center', fontsize=20,)
        self.workingTable['Text'].apply(lambda x: len(x)).value_counts(bins=10).plot(kind='bar')
        for bars in ax.containers:
            ax.bar_label(bars, fontsize=13)
        plt.xticks(rotation=45, fontsize=13)
        plt.yticks(fontsize=13)
        plt.xlabel('Sentence length', labelpad=15, fontsize=15)
        plt.ylabel('Counts', labelpad=15, fontsize=15)
        plt.show()

    def plot_labels_correlation(self):
        cormat = self.workingTable[self.label_names].corr()
        fig, ax = plt.subplots(figsize=(20, 15))
        ax.set_title('Labels correlation', loc='center', fontsize=20, )
        sns.heatmap(round(cormat, 2), annot=True)
        plt.tight_layout()
        ax.tick_params(axis='x', rotation=90)

        plt.show()
    
    def get_target_cols(self):
        temp_target_cols = [col for col in self.workingTable.columns if col not in ['Argument ID', 'Text']]
        if not SINGLE_CLASS:
            self.target_cols = temp_target_cols
        else:
            self.target_cols = [temp_target_cols[CLASS]]
        return self.target_cols


if __name__ == '__main__':
    print('Data Loader Util')
    dl = Loader()
    dl.load(clean=False, w_sep=True, w_concl=True, w_stance=False)
    dl.plot_labels_correlation()
    #dl.plot_sentences_length()

    #dl.show_label_stats()
    #max_len, sentence_max_len = dl.get_max_len()
    #print('max_len: ' + str(max_len) + ' Sentence: ' + sentence_max_len)