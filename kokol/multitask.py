import math
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
from common import common
from common.multitask import Task, TaskLayer, MultiTaskModel
from common import tensorboard as tfboard
from transformers import AutoTokenizer, TrainingArguments, Trainer
from functools import partial
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
pretrained_model_name = "roberta-base"
pretrained_model_name2 = "bert-base-uncased"
# pretrained_model_name = "bert-large-uncased"
# pretrained_model_name = "facebook/bart-base"
perform_train         = True
perform_evaluation    = False
perform_test          = True
learning_rate         = 2e-5
# learning_rate         = 3e-3
batch_size            = 16
metric_name           = "loss"
num_train_epochs      = 16
use_data_augmentation = True
include_validation_for_train = True
use_class_weights     = True
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

datadir = '../Data'
#df_train_vast, df_validation_vast, df_test_vast = common.getData(datadir + "/vast", True)
df_train, df_validation, df_test = common.getData(datadir)

from sklearn.model_selection import train_test_split

df_with_val = pd.concat([df_train, df_validation], ignore_index=True)


#df_train_vast = df_train_vast.dropna()
#df_train = pd.concat([df_train, df_train_vast])

df_train_1, df_train_2 = train_test_split(df_train, test_size=0.3)
df_train_with_val, df_val_with_val = train_test_split(df_with_val, test_size=0.03)
# Make sure row indexes are re-generated.
df_train_1.reset_index(drop=True)
df_train_2.reset_index(drop=True)
df_train_with_val.reset_index(drop=True)
df_val_with_val.reset_index(drop=True)

df_train_copy = df_train_with_val.copy()
#Augment using wordnet
tmp_aug = df_train_copy.copy()
aug = naw.SynonymAug(aug_src='wordnet', lang='eng')
print("wordnet")
for i, row in tmp_aug.iterrows():
    #print(tmp_aug.at[i,'P+S+C'])
    if i % 10 == 0:
        print(i)
    new_text=aug.augment(tmp_aug.at[i,'P+S+C'])
    j=0
    #for j in range(0,2):
    tmp_aug.at[i,'P+S+C'] = str(new_text[j])
df_train_with_val = pd.concat([df_train_with_val, tmp_aug])
'''
#augment using BERT
print("BERT")
tmp_aug = df_train_copy.copy()
aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', aug_p=0.2, device='cuda')
for i, row in tmp_aug.iterrows():
    if i%10==0:
        print(i)
    new_text=aug.augment(tmp_aug.at[i,'P+S+C'])
    tmp_aug.at[i,'P+S+C'] = new_text[0]
df_train = pd.concat([df_train, tmp_aug])
'''
df_train = df_train.sample(frac=1)

# Get dataset labels...
#labels = [label for label in dataset['train'].features.keys() if label not in common.dataLabels]
labels = [label for label in df_train.columns if label not in common.dataLabels]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}
# print("Labels:", len(labels), labels)
tfboard.display_labels=labels

# Maria's idea...
#df_train = common.remove_noisy_examples(df_train, labels=labels, classes=None)
dataset1 = common.getDatasets(df_train_1, df_train_2, df_test)
dataset = common.getDatasets(df_train, df_validation, df_test)
datasetval = common.getDatasets(df_train_with_val, df_val_with_val, df_test)

# Get class weights...
loss_pos_weights   = None
loss_class_weights = None

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
    loss_class_weights = common.compute_class_weights2(pd.concat([df_train, df_validation], ignore_index=True, sort=False), labels)
    print("Class weights: sum()=", sum(loss_class_weights))
    for i, lbl in enumerate(labels):
        print(lbl, "=", loss_class_weights[i])

if use_pos_weights:
    loss_pos_weights = common.compute_positive_weights(pd.concat([df_train, df_validation], ignore_index=True, sort=False), labels)
    print("Positive weights: sum()=", sum(loss_pos_weights))
    for i, lbl in enumerate(labels):
        print(lbl, "=", loss_pos_weights[i])

args_pretrain = TrainingArguments(
    output_dir                  = output_dir,
    evaluation_strategy         = "epoch",
    save_strategy               = "epoch" if save_checkpoints else "no",
    eval_steps                  = 10,
    learning_rate               = learning_rate,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size  = 8,
    num_train_epochs            = 5,
    weight_decay                = 0.01,
    load_best_model_at_end      = True if save_checkpoints else False,
    metric_for_best_model       = metric_name,
    seed                        = seed,
    # push_to_hub=True,
)

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
    #TaskLayer(layer_type="Conv1d", in_channels=1, out_channels=1, kernel_size=3, padding=1),
    #TaskLayer(layer_type="MaxPool1d", kernel_size=2),
    #TaskLayer(out_features=128, activation="SiLU", dropout_p=0.1),

    #TaskLayer(layer_type="ResStart"),
    #TaskLayer(layer_type="Conv1d", in_channels=64, out_channels=64, kernel_size=3, padding=3, stride=2),
    #TaskLayer(layer_type="Conv1d", in_channels=64, out_channels=1, kernel_size=3, padding=1, stride=1),
    #TaskLayer(layer_type="ResEnd"),
    TaskLayer(out_features=64, activation="SiLU", dropout_p=0.1),
]
task_layers2 = [
    #TaskLayer(layer_type="Conv1d", in_channels=1, out_channels=1, kernel_size=5, padding=2),
    #TaskLayer(layer_type="AvgPool1d", kernel_size=2),
    #TaskLayer(out_features=128, activation="SiLU", dropout_p=0.1),
    #TaskLayer(out_features=128, activation="SiLU", dropout_p=0.1),
    #TaskLayer(out_features=128, activation="SiLU", dropout_p=0.1),
    TaskLayer(out_features=64, activation="SiLU", dropout_p=0.1),

]

tid = 0
tasks = [
    Task(id=(tid := tid + 1), type="seq_classification_siamese",name="v-CE-mean", num_labels=len(labels), problem_type="multi_label_classification",
         loss="CrossEntropyLoss", loss_reduction="mean", loss_pos_weight=loss_pos_weights,
         loss_class_weight=loss_class_weights, task_layers=task_layers, task_layers2=task_layers2),
    Task(id=(tid := tid + 1),type="seq_classification_siamese", name="v-BCE-mean", num_labels=len(labels), problem_type="multi_label_classification",
         loss="BCEWithLogitsLoss", loss_reduction="mean", loss_pos_weight=loss_pos_weights,
         loss_class_weight=loss_class_weights, task_layers=task_layers, task_layers2=task_layers2),
    # Task(id=(tid:=tid+1), name="v-CE-mean-minorities", num_labels=len(labels),   problem_type="multi_label_classification", loss="CrossEntropyLoss",         loss_reduction="mean",  loss_pos_weight=loss_pos_weights, loss_class_weight=loss_class_weights, task_layers=task_layers),
    # Task(id=(tid:=tid+1), name="v-BCE-mean-minorities", num_labels=len(labels),  problem_type="multi_label_classification", loss="BCEWithLogitsLoss",        loss_reduction="mean",  loss_pos_weight=loss_pos_weights, loss_class_weight=loss_class_weights, task_layers=task_layers),
    Task(id=(tid := tid + 1), type="seq_classification_siamese",name="v-CE-mean-minorities-nw", num_labels=len(labels),
         problem_type="multi_label_classification", loss="CrossEntropyLoss", loss_reduction="mean",
         loss_pos_weight=None, loss_class_weight=None, task_layers=task_layers, task_layers2=task_layers2),
    Task(id=(tid := tid + 1), type="seq_classification_siamese",name="v-BCE-mean-minorities-nw", num_labels=len(labels),
         problem_type="multi_label_classification", loss="BCEWithLogitsLoss", loss_reduction="mean",
         loss_pos_weight=None, loss_class_weight=None, task_layers=task_layers, task_layers2=task_layers2),
]
print("Task ids:", [t.id for t in tasks])
tensorbaord_filename_suffix = "_".join([t.name for t in tasks])
tfboard.filename_suffix = tensorbaord_filename_suffix




def instantiate_model(pretrained_model_name, pretrained_model_name2, tasks, freezeLayers=False):
    model = MultiTaskModel(pretrained_model_name, pretrained_model_name2, tasks)
    ## Freeze layers...
    if freezeLayers:
        model.freeze(False)
    return model

#model = instantiate_model(pretrained_model_name, tasks, freeze_layers_bert)

#forward pass
#outputs = model(input_ids=encoded_dataset['train']['input_ids'][0].unsqueeze(0))
#print(outputs)
#print(len(outputs['hidden_states']))

TRIAL_SCORES = []
def model_init(trial=None):
    common.setSeeds(seed)
    TRIAL_SCORES = []
    print("==============> model_init <=====================")
    print("trial:", trial)
    if trial is not None:
        params = trial.params
        print("Trial Params:", params)
        if "n_layers" in params:
            task_layers = []
            for l in range(params["n_layers"]):
                task_layers.append(TaskLayer(out_features=params[f"n_units_l{l}"], dropout_p=0.1))
            tasks[0].task_layers = task_layers
        if "theta" in params:
            tasks[0].loss_reduction_weight = (1. + params["theta"])**2
            tasks[1].loss_reduction_weight = (1. - params["theta"])**2


    model = instantiate_model(pretrained_model_name, pretrained_model_name2, tasks, freeze_layers_bert)
    # print(model)
    #summary(model, input_size=(2, max_length), depth=4, dtypes=['torch.IntTensor'], device="cpu")

    #print(dict(model.named_parameters()))
    #print(model.state_dict())
    #exit(0)
    return model


## Tokenise dataset...
tokenizer1 = AutoTokenizer.from_pretrained(pretrained_model_name)
tokenizer2 = AutoTokenizer.from_pretrained(pretrained_model_name)
#encoded_dataset = common.encodeDataset(dataset, labels, tokenizer, max_length, sent1="Premise", sent2="Conclusion", task_ids=[t.id for t in tasks])
encoded_dataset1 = common.encodeDataset(dataset1, labels, tokenizer1, max_length, sent1=Sentence1, sent2=Sentence2, task_ids=[t.id for t in tasks])
encoded_dataset2 = common.encodeDataset(dataset1, labels, tokenizer2, max_length, sent1=Sentence1, sent2=Sentence2, task_ids=[t.id for t in tasks])
encoded_dataset = common.encodeDataset(dataset, labels, tokenizer1, max_length, sent1=Sentence1, sent2=Sentence2, task_ids=[t.id for t in tasks])
encoded_val = common.encodeDataset(datasetval, labels, tokenizer1, max_length, sent1=Sentence1, sent2=Sentence2, task_ids=[t.id for t in tasks])

model_train = model_init()

training_count = 1
tfboard.filename_suffix = tensorbaord_filename_suffix + f"_tr{training_count}"

trainer = Trainer(
        args=args_pretrain,
        train_dataset = encoded_dataset1["train"],
        eval_dataset  = encoded_dataset1["validation"],
        #train_dataset = encoded_dataset["validation"],
        #eval_dataset  = encoded_dataset["train"],
        tokenizer=tokenizer1,
        compute_metrics=partial(common.compute_metrics, labels=labels, tasks=tasks),
        model=model_train,
        callbacks=[tfboard.MTTensorBoardCallback],
)
trainer.remove_callback(TensorBoardCallback)
#common.show_memory("Memory before Training")
print(f"############### Training: {training_count}")
trainer.train()

training_count +=1
tfboard.filename_suffix = tensorbaord_filename_suffix + f"_tr{training_count}"

trainer = Trainer(
        args=args_pretrain,
        train_dataset = encoded_dataset2["train"],
        eval_dataset  = encoded_dataset2["validation"],
        #train_dataset = encoded_dataset["validation"],
        #eval_dataset  = encoded_dataset["train"],
        tokenizer=tokenizer2,
        compute_metrics=partial(common.compute_metrics, labels=labels, tasks=tasks),
        model=model_train,
        callbacks=[tfboard.MTTensorBoardCallback],
)
trainer.remove_callback(TensorBoardCallback)
#common.show_memory("Memory before Training")
print(f"############### Training: {training_count}")
trainer.train()

training_count +=1
tfboard.filename_suffix = tensorbaord_filename_suffix + f"_tr{training_count}"

trainer = Trainer(
        args=args,
        train_dataset = encoded_dataset["train"],
        eval_dataset  = encoded_dataset["validation"],
        #train_dataset = encoded_dataset["validation"],
        #eval_dataset  = encoded_dataset["train"],
        tokenizer=tokenizer1,
        compute_metrics=partial(common.compute_metrics, labels=labels, tasks=tasks),
        model=model_train,
        callbacks=[tfboard.MTTensorBoardCallback],
)
trainer.remove_callback(TensorBoardCallback)
#common.show_memory("Memory before Training")
print(f"############### Training: {training_count}")
trainer.train()

training_count +=1
tfboard.filename_suffix = tensorbaord_filename_suffix + f"_tr{training_count}"

trainer = Trainer(
        args=args_pretrain,
        train_dataset = encoded_val["train"],
        eval_dataset  = encoded_val["validation"],
        #train_dataset = encoded_dataset["validation"],
        #eval_dataset  = encoded_dataset["train"],
        tokenizer=tokenizer1,
        compute_metrics=partial(common.compute_metrics, labels=labels, tasks=tasks),
        model=model_train,
        callbacks=[tfboard.MTTensorBoardCallback],
)
trainer.remove_callback(TensorBoardCallback)
#common.show_memory("Memory before Training")
print(f"############### Training: {training_count}")
trainer.train()

training_count +=1
tfboard.filename_suffix = tensorbaord_filename_suffix + f"_tr{training_count}"

#common.show_memory("Memory after Training")

if perform_evaluation:
    print("############### Evaluation:")
    common.save_eval_result_df = df_val_with_val
    common.evaluationResultsFilename = "validation.tsv"
    results = trainer.evaluate()
    trainer.save_model(best_output_dir)
    #common.show_memory("Memory after Evaluation")
    common.save_eval_result_df = None
    print(results)
    # cmd = subprocess.run(["python", "evaluator.py",
    #                       "--inputDataset", "evaluationLabels",
    #                       "--inputRun", "evaluationResults",
    #                       "--outputDataset", "evaluationResults"])
if perform_test:
    print("############### Test:")
    common.save_eval_result_df = df_test
    common.evaluationResultsFilename = "test.tsv"
    results = trainer.predict(encoded_dataset["test"])
    common.save_eval_result_df = None

