from common import common
from common.multitask import Task, TaskLayer, MultiTaskModel
from transformers import AutoTokenizer, TrainingArguments, Trainer
from functools import partial
import subprocess
import numpy as np
import pandas as pd

common.setSeeds(2022)

## Read dataset...
datadir = '../Data'
df_train, df_validation, df_test = common.getData(datadir)
dataset = common.getDatasets(df_train, df_validation, df_test)

## Get dataset labels...
labels = [label for label in dataset['train'].features.keys() if label not in common.dataLabels]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}
#print("Labels:", labels)
## Get class weights...
class_weights = common.compute_class_weights2(pd.concat([df_train, df_validation], ignore_index=True, sort=False), labels)
loss_pos_weights = None

############################################################
## Parameters
############################################################
pretrained_model_name = "bert-base-uncased"
#pretrained_model_name = "facebook/bart-base"
batch_size            = 96
metric_name           = "f1"
num_train_epochs      = 16
use_class_weights     = False
use_pos_weights       = True
freeze_layers_bert    = True

if use_class_weights:
    print("Class weights: sum()=", sum(class_weights))
    for i, lbl in enumerate(labels):
        print(lbl, "=", class_weights[i])
if use_pos_weights:
    loss_pos_weights = common.compute_positive_weights(pd.concat([df_train, df_validation], ignore_index=True, sort=False), labels)
    print("Positive weights: sum()=", sum(loss_pos_weights))
    for i, lbl in enumerate(labels):
        print(lbl, "=", loss_pos_weights[i])

args = TrainingArguments(
    f"mt-{pretrained_model_name}-{num_train_epochs}-{batch_size}-{metric_name}",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_train_epochs,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    #push_to_hub=True,
)
task_layers = [
    TaskLayer(out_features=256, dropout_p=0.1),
    TaskLayer(out_features=256, dropout_p=0.1),
]

tasks = [
    Task(id=0, name="values", num_labels=len(labels), problem_type="multi_label_classification", loss="BCEWithLogitsLoss", loss_reduction="sum", loss_pos_weight=loss_pos_weights, task_layers=task_layers),
    Task(id=1, name="stance", num_labels=2, problem_type="single_label_classification", loss="sigmoid_focal_loss", loss_reduction="sum", labels="labels_stance")
]
print("Task ids:", [t.id for t in tasks])

## Tokenise dataset...
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
encoded_dataset = common.encodeDataset(dataset, labels, tokenizer, 200, sent1="Premise", sent2="Conclusion", task_ids=[t.id for t in tasks])


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

def model_init(trial=None):
    print("==============> model_init <=====================")
    print("trial:", trial)
    return instantiate_model(pretrained_model_name, tasks, freeze_layers_bert)

trainer = Trainer(
        args=args,
        train_dataset = encoded_dataset["train"],
        eval_dataset  = encoded_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=partial(common.compute_metrics, labels=labels),
        model_init=model_init
)

if use_class_weights:
    trainer.class_weights(weights=class_weights)

#common.show_memory("Memory before Training")
print("############### Training:")
trainer.train()
#common.show_memory("Memory after Training")
print("############### Evaluation:")
common.save_eval_result_df = df_validation
results = trainer.evaluate()
trainer.save_model(f"mt-best-{pretrained_model_name}-{num_train_epochs}-{batch_size}-{metric_name}")
#common.show_memory("Memory after Evaluation")
common.save_eval_result_df = None
print(results)
cmd = subprocess.run(["python", "evaluator.py",
                      "--inputDataset", "evaluationLabels",
                      "--inputRun", "evaluationResults",
                      "--outputDataset", "evaluationResults"])
