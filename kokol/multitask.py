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
from torch.utils.tensorboard import SummaryWriter
from transformers.integrations import TensorBoardCallback

seed = 2022
common.setSeeds(seed)


# from transformers import AutoModelForSequenceClassification

# Read dataset...
datadir = '../Data'
df_train, df_validation, df_test = common.getData(datadir)
dataset = common.getDatasets(df_train, df_validation, df_test)

# Get dataset labels...
labels = [label for label in dataset['train'].features.keys() if label not in common.dataLabels]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}
# print("Labels:", labels)
tfboard.display_labels=labels
# Get class weights...
loss_pos_weights   = None
loss_class_weights = None

############################################################
## Parameters
############################################################
pretrained_model_name = "bert-base-uncased"
# pretrained_model_name = "bert-large-uncased"
# pretrained_model_name = "facebook/bart-base"
learning_rate         = 2e-5
# learning_rate         = 3e-3
batch_size            = 1024
metric_name           = "f1"
num_train_epochs      = 32
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

writer = SummaryWriter(tensorboard_dir)

if not freeze_layers_bert:
    if "large" in pretrained_model_name:
        batch_size = 32
    else:
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
    # TaskLayer(in_features=768, out_features=100, activation=None),
    TaskLayer(layer_type="Conv1d", in_channels=1, out_channels=1, kernel_size=5, padding=1, activation=None),
    TaskLayer(out_features=768, activation="ReLU"),
    TaskLayer(layer_type="AvgPool1d", kernel_size=3),
    TaskLayer(out_features=256, activation="ReLU"),
]

tid = 0
tasks = [
    Task(id=(tid:=tid+1), name="values", num_labels=len(labels), problem_type="multi_label_classification", loss="CrossEntropyLoss",         loss_reduction="sum",  loss_pos_weight=loss_pos_weights, loss_class_weight=loss_class_weights, task_layers=task_layers),
    Task(id=(tid:=tid+1), name="values", num_labels=len(labels), problem_type="multi_label_classification", loss="MultiLabelSoftMarginLoss", loss_reduction="sum",  loss_pos_weight=loss_pos_weights, loss_class_weight=loss_class_weights, task_layers=None),
    #Task(id=(tid:=tid+1), name="values", num_labels=len(labels), problem_type="multi_label_classification", loss="BCEWithLogitsLoss",        loss_reduction="mean", loss_pos_weight=loss_pos_weights, loss_class_weight=loss_class_weights, task_layers=None),
    #Task(id=(tid:=tid+1), name="values", num_labels=len(labels), problem_type="multi_label_classification", loss="SigmoidMultiLabelSoftMarginLoss", loss_reduction="sum", loss_pos_weight=loss_pos_weights, loss_class_weight=loss_class_weights, task_layers=None),

    #Task(id=(tid:=tid+1), name="stance", num_labels=2, problem_type="single_label_classification", loss="sigmoid_focal_loss", loss_reduction="sum", labels="labels_stance")

]
print("Task ids:", [t.id for t in tasks])

## Tokenise dataset...
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
encoded_dataset = common.encodeDataset(dataset, labels, tokenizer, max_length, sent1="Premise", sent2="Conclusion", task_ids=[t.id for t in tasks])


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
    summary(model, input_size=(2, max_length), depth=4, dtypes=['torch.IntTensor'], device="cpu")

    #print(dict(model.named_parameters()))
    #print(model.state_dict())
    #exit(0)
    return model

trainer = Trainer(
        args=args,
        train_dataset = encoded_dataset["train"],
        eval_dataset  = encoded_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=partial(common.compute_metrics, labels=labels, tasks=tasks, writer=writer),
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
