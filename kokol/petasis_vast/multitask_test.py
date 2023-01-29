from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
from common import common
from common.multitask import Task, TaskLayer, MultiTaskModel
from common import tensorboard as tfboard
from transformers import AutoTokenizer, TrainingArguments, Trainer
from functools import partial
from collections import Counter
import subprocess
import numpy as np
import pandas as pd
from torchinfo import summary
import optuna
from transformers.integrations import TensorBoardCallback

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc

import os
os.environ["MODEL_DIR"] = '/tmp/models'

############################################################
## Parameters
############################################################
seed = 2022
pretrained_model_name = "bert-base-uncased"
# pretrained_model_name = "bert-large-uncased"
# pretrained_model_name = "facebook/bart-base"
learning_rate         = 2e-5
# learning_rate         = 3e-3
batch_size            = 8
metric_name           = "f1"
num_train_epochs      = 32
use_class_weights     = False
use_pos_weights       = True
freeze_layers_bert    = False
max_length            = 200
hperparam_search      = False
save_checkpoints      = False
output_dir            = f"runs/mt-{pretrained_model_name}-{num_train_epochs}-{batch_size}-{metric_name}"
best_output_dir       = f"runs/mt-best-{pretrained_model_name}-{num_train_epochs}-{batch_size}-{metric_name}"
tensorboard_dir       = f"runs/mt-tb-{pretrained_model_name}-{num_train_epochs}-{batch_size}-{metric_name}"
hperparam_search_name = f"runs/mt-std-{pretrained_model_name}-{num_train_epochs}-{batch_size}-{metric_name}"
Sentence1             = 'P+S+C'
Sentence2             = None

common.setSeeds(seed)

# from transformers import AutoModelForSequenceClassification

# Read dataset...
datadir = '../../Data'
#df_train_vast, df_validation_vast, df_test_vast = common.getData(datadir + "/vast", True)
df_train, df_validation, df_test = common.getData(datadir)


# Get dataset labels...
#labels = [label for label in dataset['train'].features.keys() if label not in common.dataLabels]
labels = [label for label in df_train.columns if label not in common.dataLabels]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}
# print("Labels:", len(labels), labels)

# Maria's idea...
#df_train = common.remove_noisy_examples(df_train, labels=labels, classes=None)
dataset = common.getDatasets(df_train, df_validation, df_test)

# Get class weights...
loss_pos_weights   = None
loss_class_weights = None

if not freeze_layers_bert:
    if "large" in pretrained_model_name:
        if batch_size > 32:
            batch_size = 32
    else:
        if batch_size > 64:
            batch_size = 64

if use_class_weights:
    loss_class_weights = common.compute_class_weights3(pd.concat([df_train, df_validation], ignore_index=True, sort=False), labels)
    print("Class weights: sum()=", sum(loss_class_weights))
    for i, lbl in enumerate(labels):
        print(lbl, "=", loss_class_weights[i])

if use_pos_weights:
    loss_pos_weights = common.compute_positive_weights(pd.concat([df_train, df_validation], ignore_index=True, sort=False), labels)
    print("Positive weights: sum()=", sum(loss_pos_weights))
    for i, lbl in enumerate(labels):
        print(lbl, "=", loss_pos_weights[i])

args = TrainingArguments(
    output_dir                  = output_dir,
    evaluation_strategy         = "epoch",
    save_strategy               = "epoch" if save_checkpoints else "no",
    eval_steps                  = 10,
    learning_rate               = learning_rate,
    per_device_train_batch_size = batch_size,
    per_device_eval_batch_size  = batch_size,
    num_train_epochs            = num_train_epochs,
    weight_decay                = 0.01,
    load_best_model_at_end      = True if save_checkpoints else False,
    metric_for_best_model       = metric_name,
    seed                        = seed,
    # push_to_hub=True,
)

task_layers = [
    TaskLayer(out_features=384, dropout_p=0.1, activation="ReLU"),
]
task_layers=None

tid = 0
tasks = [
    Task(id=(tid:=tid+1), name="v-CE-mean", num_labels=len(labels),   problem_type="multi_label_classification", loss="CrossEntropyLoss",         loss_reduction="mean",  loss_pos_weight=loss_pos_weights, loss_class_weight=loss_class_weights, task_layers=task_layers),
    Task(id=(tid:=tid+1), name="v-BCE-mean", num_labels=len(labels),  problem_type="multi_label_classification", loss="BCEWithLogitsLoss",        loss_reduction="mean",  loss_pos_weight=loss_pos_weights, loss_class_weight=loss_class_weights, task_layers=task_layers),
    #Task(id=(tid:=tid+1), name="v-CE-mean-minorities", num_labels=len(labels),   problem_type="multi_label_classification", loss="CrossEntropyLoss",         loss_reduction="mean",  loss_pos_weight=loss_pos_weights, loss_class_weight=loss_class_weights, task_layers=task_layers),
    #Task(id=(tid:=tid+1), name="v-BCE-mean-minorities", num_labels=len(labels),  problem_type="multi_label_classification", loss="BCEWithLogitsLoss",        loss_reduction="mean",  loss_pos_weight=loss_pos_weights, loss_class_weight=loss_class_weights, task_layers=task_layers),
    Task(id=(tid:=tid+1), name="v-CE-mean-minorities-nw", num_labels=len(labels),   problem_type="multi_label_classification", loss="CrossEntropyLoss",         loss_reduction="mean",  loss_pos_weight=None, loss_class_weight=None, task_layers=task_layers),
    Task(id=(tid:=tid+1), name="v-BCE-mean-minorities-nw", num_labels=len(labels),  problem_type="multi_label_classification", loss="BCEWithLogitsLoss",        loss_reduction="mean",  loss_pos_weight=None, loss_class_weight=None, task_layers=task_layers),

    #Task(id=(tid:=tid+1), name="v-MLSM-mean", num_labels=len(labels), problem_type="multi_label_classification", loss="MultiLabelSoftMarginLoss", loss_reduction="mean",  loss_pos_weight=loss_pos_weights, loss_class_weight=loss_class_weights, task_layers=task_layers),
    #Task(id=(tid:=tid+1), name="v-SF-mean", num_labels=len(labels),   problem_type="multi_label_classification", loss="sigmoid_focal_loss",       loss_reduction="mean")
    #Task(id=(tid:=tid+1), name="values", num_labels=len(labels), problem_type="multi_label_classification", loss="SigmoidMultiLabelSoftMarginLoss", loss_reduction="sum", loss_pos_weight=loss_pos_weights, loss_class_weight=loss_class_weights, task_layers=None),
    #Task(id=(tid:=tid+1), name="stance", num_labels=2, problem_type="single_label_classification", loss="sigmoid_focal_loss", loss_reduction="sum", labels="labels_stance")

]
print("Task ids:", [t.id for t in tasks])
minority_task_ids = [3, 4, 5, 6]
tfboard.filename_suffix = "_".join([t.name for t in tasks])

## Tokenise dataset...
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
encoded_dataset = common.encodeDataset(dataset, labels, tokenizer, max_length, sent1=Sentence1, sent2=Sentence2, task_ids=[t.id for t in tasks])

if minority_task_ids is not None and len(minority_task_ids):
    # Calculate minority classes...
    counter = Counter()
    for i, c in enumerate(labels):
        counter[i] += df_train[c].sum()
    minority_classes = []
    minority_class_indexes = []
    for c, f in counter.most_common():
        if f < 850:
            minority_class_indexes.append(c)
            minority_classes.append(labels[c])
            print(f"Minority Class: {labels[c]} ({f})")

    encoded_dataset['train'] = common.split_imballance_dataset(encoded_dataset['train'],
                                                               labels=labels,
                                                               minority_class_indexes=minority_class_indexes,
                                                               task_ids=[t.id for t in tasks],
                                                               minority_task_ids=minority_task_ids,
                                                               random_percent=0.1)

def instantiate_model(pretrained_model_name, tasks, freezeLayers=False):
    model = MultiTaskModel(pretrained_model_name, tasks)
    ## Freeze layers...
    model.eval()
    return model

#model = instantiate_model(pretrained_model_name, tasks, freeze_layers_bert)

#forward pass
#outputs = model(input_ids=encoded_dataset['train']['input_ids'][0].unsqueeze(0))
#print(outputs)
#print(len(outputs['hidden_states']))


#common.show_memory("Memory before Training")
print("############### Training:")




#common.show_memory("Memory after Evaluation")
common.save_eval_result_df = None
print(results)
cmd = subprocess.run(["python", "evaluator.py",
                      "--inputDataset", "evaluationLabels",
                      "--inputRun", "evaluationResults",
                      "--outputDataset", "evaluationResults"])
