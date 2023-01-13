import random
import numpy as np
import torch
import pandas as pd
from datasets import Dataset, DatasetDict
from functools import partial
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, precision_recall_fscore_support, fbeta_score
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from transformers import EvalPrediction
from collections import Counter
import os
import csv

def setSeeds(seed=2022):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def show_memory(text=''):
  t = torch.cuda.get_device_properties(0).total_memory
  c = torch.cuda.memory_cached(0)
  a = torch.cuda.memory_allocated(0)
  f = c-a  # free inside cache
  print(f'\n\n{text}\nTotal: {t}\nCached: {c} \nAllocated: {a} \nFree in cache: {f}\n\n')

dataLabels = ['Argument ID', 'Conclusion', 'Stance', 'Premise', '__index_level_0__', 'P+S', 'C+S']
def getData(datadir):
    df_args = pd.read_csv(datadir + '/arguments-training.tsv', sep = '\t')
    df_args['P+S'] = df_args[['Premise',    'Stance']].apply(lambda x: ' '.join(x), axis=1)
    df_args['C+S'] = df_args[['Conclusion', 'Stance']].apply(lambda x: ' '.join(x), axis=1)
    df_lbls = pd.read_table(datadir + '/labels-training.tsv')
    df_train = df_args.merge(df_lbls, how="left", on="Argument ID")

    df_args = pd.read_csv(datadir + '/arguments-validation.tsv', sep = '\t')
    df_args['P+S'] = df_args[['Premise',    'Stance']].apply(lambda x: ' '.join(x), axis=1)
    df_args['C+S'] = df_args[['Conclusion', 'Stance']].apply(lambda x: ' '.join(x), axis=1)
    df_lbls = pd.read_table(datadir + '/labels-validation.tsv')
    df_validation = df_args.merge(df_lbls, how="left", on="Argument ID")

    df_args = pd.read_table(datadir + '/arguments-test.tsv')
    df_args['P+S'] = df_args[['Premise',    'Stance']].apply(lambda x: ' '.join(x), axis=1)
    df_args['C+S'] = df_args[['Conclusion', 'Stance']].apply(lambda x: ' '.join(x), axis=1)
    df_test = df_args

    return df_train, df_validation, df_test

def getDatasets(df_train, df_validation, df_test):
    train_dataset      = Dataset.from_pandas(df_train,      split="train")
    validation_dataset = Dataset.from_pandas(df_validation, split="validation")
    test_dataset       = Dataset.from_pandas(df_test,       split="test")
    dataset            = DatasetDict({ "train": train_dataset, "validation": validation_dataset, "test": test_dataset })
    return dataset

def preprocess_data(examples, labels, tokenizer, max_length=200):
    # take a batch of texts
    premise    = examples["Premise"]
    # conclusion = examples["Conclusion"]
    conclusion = examples["C+S"]
    # stance     = examples["Stance"]
    # encode them
    encoding = tokenizer(conclusion, premise, padding="max_length", truncation=True, max_length=max_length)
    # add labels
    labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
    ## Test may not have labels...
    if (len(labels_batch)):
        # create numpy array of shape (batch_size, num_labels)
        labels_matrix = np.zeros((len(premise), len(labels)))
        # fill numpy array
        for idx, label in enumerate(labels):
            labels_matrix[:, idx] = labels_batch[label]

        encoding["labels"] = labels_matrix.tolist()
    else:
        labels_matrix = np.zeros((len(premise), len(labels)))
        encoding["labels"] = labels_matrix.tolist()

    return encoding

def encodeDataset(dataset, labels, tokenizer, max_length=200):
    encoded_dataset = dataset.map(
        partial(preprocess_data, labels=labels, tokenizer=tokenizer, max_length=max_length),
        batched=True)
    encoded_dataset['train']      = encoded_dataset['train'].remove_columns(dataset['train'].column_names)
    encoded_dataset['validation'] = encoded_dataset['validation'].remove_columns(dataset['validation'].column_names)
    encoded_dataset['test']       = encoded_dataset['test'].remove_columns(dataset['test'].column_names)
    encoded_dataset.set_format("torch")
    return encoded_dataset

# From: https://stackoverflow.com/questions/54842067/how-to-calculate-class-weights-of-a-pandas-dataframe-for-keras
def compute_class_weights(df, class_weight='balanced'):
    sdf = df = df.stack().reset_index()
    Y = sdf[df[0] == 1]['level_1']
    class_weights = compute_class_weight(
        class_weight=class_weight, classes=np.unique(Y), y=Y
    )
    return class_weights

def compute_class_weights2(df, labels):
    counter = Counter()
    for label in labels:
        counter[label] += len(df[(df[label] > 0)])
    total = sum(counter.values())
    class_weights = []
    n_classes = len(labels)
    for label in labels:
        # n_samples / (n_classes * np.bincount(y))
        class_weights.append(total/ (n_classes * counter[label]))
    # print(class_weights)
    return np.array(class_weights)

save_eval_result_df = None
def save_eval_results(predictions, labels=[], evaluationResultsDir="evaluationResults", evaluationResultsFilename="run.tsv"):
    if (save_eval_result_df is None):
        return
    # make sure the shapes match...
    if len(save_eval_result_df.index) != len(predictions):
        raise Exception("Dataframe number of rows differs from number of predictions")
    if not os.path.exists(evaluationResultsDir):
        os.makedirs(evaluationResultsDir)
    with open(os.path.join(evaluationResultsDir, evaluationResultsFilename), "w") as runFile:
        writer = csv.DictWriter(runFile, fieldnames = ["Argument ID"] + labels, delimiter = "\t")
        writer.writeheader()
        for index, row in save_eval_result_df.iterrows():
            r = { "Argument ID": row[0] }
            i = 0
            for lbl in labels:
                r[lbl] = int(predictions[index][i])
                i += 1
            writer.writerow(r)

metrics = {
    'p': [],
    'r': [],
    'f1': [],
    'f1_m': []
}
# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, true_labels, threshold=0.5, labels=[]):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # Save results...
    save_eval_results(y_pred, labels=labels)
    # finally, compute metrics
    y_true = true_labels
    precision, recall, f1, _ = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average='macro')
    #precision_micro_average = precision_score(y_true=y_true, y_pred=y_pred, average='micro')
    #recall_micro_average    = recall_score   (y_true=y_true, y_pred=y_pred, average='micro')
    #f1_micro_average        = f1_score       (y_true=y_true, y_pred=y_pred, average='micro')
    #precision_macro_average = precision_score(y_true=y_true, y_pred=y_pred, average='macro')
    #recall_macro_average    = recall_score   (y_true=y_true, y_pred=y_pred, average='macro')
    #f1_macro_average        = f1_score       (y_true=y_true, y_pred=y_pred, average='macro')
    #roc_auc                 = roc_auc_score(y_true, y_pred, average = 'micro')
    #accuracy                = accuracy_score(y_true, y_pred)
    mcm = multilabel_confusion_matrix(y_true=y_true, y_pred=y_pred, labels=np.arange(len(labels)))
    # return as dictionary
    metrics = {'p': precision,
               'r': recall,
               'f1':  2 * precision * recall / (precision + recall),
               'f1_m': f1
               # 'roc_auc': roc_auc,
               # 'accuracy': accuracy,
               # 'mcm': mcm.tolist()
              }
    return metrics

def compute_metrics(p: EvalPrediction, labels=[]):
    preds = p.predictions[0] if isinstance(p.predictions,
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds,
        true_labels=p.label_ids,
        labels=labels)
    return result
