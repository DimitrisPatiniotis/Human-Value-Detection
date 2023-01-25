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
import math


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


dataLabels = ['Argument ID', 'Conclusion', 'Stance', 'Premise', '__index_level_0__', 'P+S', 'C+S', 'stance_boolean']


def getData(datadir):
    df_args = pd.read_csv(datadir + '/arguments-training.tsv', sep = '\t')
    df_args['P+S'] = df_args[['Premise',    'Stance']].apply(lambda x: ' '.join(x), axis=1)
    df_args['C+S'] = df_args[['Conclusion', 'Stance']].apply(lambda x: ' '.join(x), axis=1)
    df_args['stance_boolean'] = df_args['Stance'].map({"against": 0, "in favor of": 1, "in favour of": 1})
    if df_args['stance_boolean'].isnull().values.any():
        raise Exception(f"NAN problem in data 1")
    df_lbls = pd.read_table(datadir + '/labels-training.tsv')
    df_train = df_args.merge(df_lbls, how="left", on="Argument ID")

    df_args = pd.read_csv(datadir + '/arguments-validation.tsv', sep = '\t')
    df_args['P+S'] = df_args[['Premise',    'Stance']].apply(lambda x: ' '.join(x), axis=1)
    df_args['C+S'] = df_args[['Conclusion', 'Stance']].apply(lambda x: ' '.join(x), axis=1)
    df_args['stance_boolean'] = df_args['Stance'].map({"against": 0, "in favor of": 1, "in favour of": 1})
    if df_args['stance_boolean'].isnull().values.any():
        raise Exception(f"NAN problem in data 1")
    df_lbls = pd.read_table(datadir + '/labels-validation.tsv')
    df_validation = df_args.merge(df_lbls, how="left", on="Argument ID")
    #print(df_validation[['Argument ID', "stance_boolean", "Stance"]].to_string())
    #exit(0)

    df_args = pd.read_table(datadir + '/arguments-test.tsv')
    df_args['P+S'] = df_args[['Premise',    'Stance']].apply(lambda x: ' '.join(x), axis=1)
    df_args['C+S'] = df_args[['Conclusion', 'Stance']].apply(lambda x: ' '.join(x), axis=1)
    df_args['stance_boolean'] = df_args['Stance'].map({"against": 0, "in favor of": 1, "in favour of": 1})
    if df_args['stance_boolean'].isnull().values.any():
        raise Exception(f"NAN problem in data 1")
    df_test = df_args

    return df_train, df_validation, df_test


def getDatasets(df_train, df_validation, df_test):
    train_dataset      = Dataset.from_pandas(df_train,      split="train")
    validation_dataset = Dataset.from_pandas(df_validation, split="validation")
    test_dataset       = Dataset.from_pandas(df_test,       split="test")
    dataset            = DatasetDict({ "train": train_dataset, "validation": validation_dataset, "test": test_dataset })
    return dataset


def preprocess_data(examples, labels, tokenizer, max_length=200, task_ids=[0], sent1="C+S", sent2="Premise"):
    # take a batch of texts
    sentA = examples[sent1]
    # conclusion = examples["Conclusion"]
    sentB = examples[sent2]
    # stance     = examples["Stance"]
    # encode them
    encoding = tokenizer(sentA, sentB, padding="max_length", truncation=True, max_length=max_length)
    # add labels
    labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
    # Test may not have labels...
    if (len(labels_batch)):
        # create numpy array of shape (batch_size, num_labels)
        labels_matrix = np.zeros((len(sentA), len(labels)))
        # fill numpy array
        for idx, label in enumerate(labels):
            labels_matrix[:, idx] = labels_batch[label]

        encoding["labels"] = labels_matrix.tolist()
    else:
        labels_matrix = np.zeros((len(sentA), len(labels)))
        encoding["labels"] = labels_matrix.tolist()
    # Is it a multitask run?
    if len(task_ids) > 0:
        encoding["labels_stance"] = examples["stance_boolean"]  # Interpreted as class indices...
    # encoding["task_ids"] = [task_ids[0]] * len(encoding["labels"])

    return encoding


def encodeDataset(dataset, labels, tokenizer, max_length=200, sent1="C+S", sent2="Premise", task_ids=[0]):
    encoded_dataset = dataset.map(
        partial(preprocess_data, labels=labels, tokenizer=tokenizer, max_length=max_length,
        sent1=sent1, sent2=sent2, task_ids=task_ids),
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
        class_weights.append(total / (n_classes * counter[label]))
        # class_weights.append(total/ counter[label])
    # print(class_weights)
    return torch.special.softmax(torch.from_numpy(np.array(class_weights)), 0)


def compute_positive_weights(df, labels):
    counter = Counter()
    for label in labels:
        counter[label] += len(df[(df[label] > 0)])
    total = len(df.index)
    pos_weights = []
    for label in labels:
        pos_weights.append(total / counter[label])
    return torch.from_numpy(np.array(pos_weights))


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
        writer = csv.DictWriter(runFile, fieldnames=["Argument ID"] + labels, delimiter="\t")
        writer.writeheader()
        for index, row in save_eval_result_df.iterrows():
            r = {"Argument ID": row[0]}
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


def multi_label_metrics_do(y_true, y_pred, labels=None, prefix="", per_class=False):
    precision, recall, f1_m, _ = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
    f1 = 2 * precision * recall / (precision + recall)
    if math.isnan(f1):
        f1 = 0.0
    mcm = multilabel_confusion_matrix(y_true=y_true, y_pred=y_pred, labels=np.arange(len(labels)))
    result = {
        f'{prefix}p':   precision,
        f'{prefix}r':   recall,
        f'{prefix}f1':  f1,
        f'{prefix}mcm': mcm
        # f'{prefix}f1_m': f1_m
    }
    # Per class...
    if per_class:
        c_p, c_r, c_f1, _ = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, average=None, zero_division=0)
        for i, v in enumerate(c_p):
            result[f'{prefix}z_p{i+1}-{labels[i]}'] = v
        for i, v in enumerate(c_r):
            result[f'{prefix}z_r{i+1}-{labels[i]}'] = v
        for i, v in enumerate(c_f1):
            result[f'{prefix}z_f{i+1}-{labels[i]}'] = v

    return result


# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, true_labels, labels=[], tasks=None, writer=None):
    ## We assume that the main task is the first task, in case of multitasking...
    # y_true = true_labels[0] if isinstance(true_labels, tuple) else true_labels
    y_true, y_true_stance = true_labels

    y_pred = torch.from_numpy(np.zeros(predictions[0].shape + (len(predictions),)))
    task_metrics = {}
    for i, task in enumerate(tasks):
        preds = predictions[i]
        match task.loss:
            case "CrossEntropyLoss":
                activation = "softmax"
            case "MultiLabelSoftMarginLoss":
                activation = "softmax"
            case _:
                activation = "sigmoid"
        match activation:
            case "sigmoid":
                # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
                sigmoid = torch.nn.Sigmoid()
                probs = sigmoid(torch.Tensor(preds))
                threshold = 0.5
            case "softmax":
                # Apply softmax only if it has not already been applied.
                s = sum(preds[0])
                if s > 1.001 or s < 0.999:
                    # print("===========> Applying softmax:", s)
                    softmax = torch.nn.Softmax(dim=1)
                    probs = softmax(torch.Tensor(preds))
                else:
                    probs = torch.Tensor(preds)
                threshold = 1. / len(labels)

        # next, use threshold to turn them into integer predictions
        indices = np.where(probs >= threshold)
        indices += (np.full( len(indices[0]), i),)
        y_pred[indices] = 1
        task_y_pred = np.zeros(probs.shape)
        task_y_pred[np.where(probs >= threshold)] = 1
        # task metrics...
        tm = multi_label_metrics_do(y_true=y_true, y_pred=task_y_pred, prefix=f"t{i+1}_", labels=labels, per_class=False)
        task_metrics = task_metrics | tm
        # print(f"Task {i+1}:", tm)
    # Implement voting (take mode)...
    y_pred = y_pred.mode(dim=-1)[0].numpy()

    # Save results...
    save_eval_results(y_pred, labels=labels)

    # finally, compute metrics
    metrics = multi_label_metrics_do(y_true=y_true, y_pred=y_pred, labels=labels, per_class=True) | task_metrics
    return metrics

def compute_metrics(p: EvalPrediction, labels=[], tasks=None, writer=None):
    #print("predictions:", p.predictions)
    #print("labels:", p.label_ids)
    #preds = p.predictions[0] if isinstance(p.predictions,
    #        tuple) else p.predictions
    # lbls  = p.label_ids[0] if isinstance(p.label_ids,
    #         tuple) else p.label_ids
    # print("p.predictions:", p.predictions, len(p.predictions), type(p.predictions), type(p.predictions[0]), type(p.predictions[1]))
    # print("preds:", preds, type(preds))
    result = multi_label_metrics(
        predictions=p.predictions,
        true_labels=p.label_ids,
        labels=labels,
        tasks=tasks,
        writer=writer)
    return result
