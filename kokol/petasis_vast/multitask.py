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

df_train = pd.concat([df_train, df_validation], ignore_index=True)

#df_train_new=df_train.loc[(df_train['Stimulation']==1) | (df_train['Hedonism']==1) | (df_train['Face']==1)]
#df_train=pd.concat([df_train.sample(n=math.floor(df_train_new.shape[0]/3)), df_train_new])

#df_train_vast = df_train_vast.dropna()
#df_train = pd.concat([df_train, df_train_vast])

df_train_copy = df_train.copy()


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
df_train = pd.concat([df_train, tmp_aug])
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

#df_train.to_csv("../../Data/arguments_augmented-training.tsv", sep='\t')

#dataset = common.getDatasets(df_train_vast, df_validation, df_test)


# Get dataset labels...
#labels = [label for label in dataset['train'].features.keys() if label not in common.dataLabels]
labels = [label for label in df_train.columns if label not in common.dataLabels]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}
# print("Labels:", len(labels), labels)
tfboard.display_labels=labels

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


    model = instantiate_model(pretrained_model_name, tasks, freeze_layers_bert)
    # print(model)
    #summary(model, input_size=(2, max_length), depth=4, dtypes=['torch.IntTensor'], device="cpu")

    #print(dict(model.named_parameters()))
    #print(model.state_dict())
    #exit(0)
    return model

trainer = Trainer(
        args=args,
        train_dataset = encoded_dataset["train"],
        eval_dataset  = encoded_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=partial(common.compute_metrics, labels=labels, tasks=tasks),
        model_init=model_init,
        callbacks=[tfboard.MTTensorBoardCallback],
)
trainer.remove_callback(TensorBoardCallback)

#####################################################################################
### Optuna Hyperparameter search
#####################################################################################

def optuna_hp_space(trial):
    # space = {
    #     "learning_rate": trial.suggest_float("learning_rate", 1e-6, 1, log=True),
    #     "n_layers": 1,
    #     "n_units_l0": 256,
    #     "n_units_l1": 256,
    #     "n_units_l2": 256,
    #     "n_units_l3": 256,
    # }
    # return space
    n_layers_min = 1
    n_layers_max = 1
    space = {
        #"n_layers": trial.suggest_int('n_layers', n_layers_min, n_layers_max),
        "theta": trial.suggest_float("theta", 0, 1)
        #"learning_rate": trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True),
        #"per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 16, 32, 64, 96]),
    }
    for i in range(n_layers_min, n_layers_max+1):
        space[f"n_units_l{i}"] = 0

    #for i in range(space["n_layers"]):
    #    space[f"n_units_l{i}"] = trial.suggest_int(f"n_units_l{i}", 128, 1024, 128)
    return space

def compute_objective(metrics: Dict[str, float]) -> float:
    # print("===========================================================================")
    # print("==== metrics:", metrics)
    # print(f"==== Objective: eval_{metric_name}:", metrics[f"eval_{metric_name}"])
    # print("===========================================================================")
    TRIAL_SCORES.append(metrics[f"eval_{metric_name}"])
    if int(metrics["epoch"]) == num_train_epochs:
        return max(TRIAL_SCORES)
    return metrics[f"eval_{metric_name}"]

if hperparam_search:
    study_name = hperparam_search_name  # Unique identifier of the study.
    storage_name = f"sqlite:///{study_name}.db"
    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=optuna_hp_space,
        compute_objective=compute_objective,
        n_trials=100,
        study_name=study_name,
        storage=storage_name,
        load_if_exists = True
    )
    print("best_trial:", best_trial)
    exit(0)

#common.show_memory("Memory before Training")
print("############### Training:")
trainer.train()
#common.show_memory("Memory after Training")
print("############### Evaluation:")
common.save_eval_result_df = df_validation
results = trainer.evaluate()
trainer.save_model(best_output_dir)
#common.show_memory("Memory after Evaluation")
common.save_eval_result_df = None
print(results)
cmd = subprocess.run(["python", "evaluator.py",
                      "--inputDataset", "evaluationLabels",
                      "--inputRun", "evaluationResults",
                      "--outputDataset", "evaluationResults"])
