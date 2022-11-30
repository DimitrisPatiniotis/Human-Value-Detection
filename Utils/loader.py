import pandas as pd
from time import time
import re
import json
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize 

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

    def __init__(self, data_base_path = '../Data/', ta_file_name='arguments-training.tsv', tl_file_name='labels-training.tsv', tl_lvl1_file_name='level1-labels-training.tsv', workingTableType='value_categories'):
        self.TA_FILE_NAME = ta_file_name
        self.TL_FILE_NAME = tl_file_name
        self.TL_LVL1_FILE_NAME = tl_lvl1_file_name
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

        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_val = None
        self.y_val = None

    def load(self):
        self.arguments = load_data(self.DATA_PATH + self.TA_FILE_NAME)
        if self.workingTableType == 'value_categories':
            self.labels = load_data(self.DATA_PATH + self.TL_FILE_NAME)
            # Merging
            self.workingTable = pd.merge(self.arguments, self.labels, on='Argument ID')
            self.label_names = get_main_label_names('../Data/value-categories.json')
        elif self.workingTableType == '':
            self.labels = load_data(self.DATA_PATH + self.TL_LVL1_FILE_NAME)
            self.workingTable = pd.merge(self.arguments, self.labels, on='Argument ID')
        # Add padding
        self.workingTable['Stance'] = self.workingTable['Stance'].apply(lambda txt : ' '+txt+' ')
        self.workingTable['Text'] = self.workingTable['Conclusion'] + self.workingTable['Stance'] + self.workingTable['Premise']
        self.workingTable = self.workingTable.drop(['Conclusion', 'Stance', 'Premise'], axis=1)
        self.workingTable['Text'] = self.workingTable['Text'].apply(clean_text)
        print(self.workingTable.head(5))
    
    def tknz(self):
        self.workingTable['Text'] = self.workingTable['Text'].apply(print, word_tokenize)
    
    def stem(self):
        self.workingTable['Text'] = self.workingTable['Text'].apply(analytical_stem)
    
    def train_test_validate_split(self):
        X, y = self.workingTable['Text'], self.workingTable[self.label_names]
        # 0.6/0.2/0.2
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        # self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.x_train, self.y_train, test_size=0.25, random_state=1)
    
    def split_to_train_val(self):
        self.workingTable = self.workingTable.sample(frac=1).reset_index(drop=True)
        self.validation = self.workingTable[1:round(len(self.workingTable) * 0.3)]
        self.train = self.workingTable[round(len(self.workingTable) * 0.3):]
    
    def show_label_stats(self):
        label_names = get_main_label_names('../Data/value-categories.json')
        total_count = []
        for label in label_names:
            total_count.append(self.workingTable[label].value_counts()[1])        
        fig, ax = plt.subplots(figsize =(16, 9))
        # Horizontal Bar Plot
        ax.barh(label_names, total_count)
        for s in ['top', 'bottom', 'left', 'right']:
            ax.spines[s].set_visible(False)
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_tick_params(pad = 5)
        ax.yaxis.set_tick_params(pad = 10)
        ax.grid(b = True, color ='grey', linestyle ='-.', linewidth = 0.5, alpha = 0.2)
        ax.invert_yaxis()
        for i in ax.patches:
            plt.text(i.get_width()+0.2, i.get_y()+0.5, str(round((i.get_width()), 2)), fontsize = 10, fontweight ='bold', color ='grey')
        ax.set_title('Occurances of Value Categories', loc ='left')
        plt.show()
    
    def get_target_cols(self):
        return [col for col in self.workingTable.columns if col not in ['Argument ID', 'Text']]



if __name__ == '__main__':
    print('Data Loader Util')
    dl = Loader()
    dl.load()
    dl.show_label_stats()