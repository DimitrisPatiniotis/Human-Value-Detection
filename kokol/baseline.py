from common import common
from transformers import AutoTokenizer
from transformers import AutoModel, AutoModelForSequenceClassification, AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer
from common.trainers import CustomTrainer
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

############################################################
## Parameters
############################################################
pretrained_model_name = "bert-base-uncased"
# pretrained_model_name = "distilbert-base-uncased"
batch_size            = 32
metric_name           = "loss"
num_train_epochs      = 130
use_class_weights     = False
freeze_layers_bert    = False

if use_class_weights:
    print("Class weights: sum()=", sum(class_weights))
    for i, lbl in enumerate(labels):
        print(lbl, "=", class_weights[i])

args = TrainingArguments(
    f"{pretrained_model_name}-finetuned-sem_eval-english",
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

## Tokenise dataset...
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
encoded_dataset = common.encodeDataset(dataset, labels, tokenizer, 200)

# input_ids = np.unique(encoded_dataset["train"][:]["input_ids"]).tolist() + \
#             np.unique(encoded_dataset["validation"][:]["input_ids"]).tolist() + \
#             np.unique(encoded_dataset["test"][:]["input_ids"]).tolist()
# unique_input_ids = np.unique(input_ids)
#
# print(unique_input_ids, len(unique_input_ids))

def instantiate_model(pretrained_model_name, freezeLayers=False):
    model = AutoModelForSequenceClassification.from_pretrained(
    # model = AutoModelForTokenClassification.from_pretrained(
    # model = AutoModel.from_pretrained(
                pretrained_model_name,
                problem_type="multi_label_classification",
                output_hidden_states=False,
                num_labels=len(labels),
                id2label=id2label,
                label2id=label2id)
    ## Freeze layers...
    if freezeLayers:
        for name, param in model.named_parameters():
            if name.startswith(("bert.embeddings", "bert.encoder")):
                param.requires_grad = False
            #print(name, param.requires_grad)
    return model

model = instantiate_model(pretrained_model_name, freeze_layers_bert)
# print(model)

#forward pass
#outputs = model(input_ids=encoded_dataset['train']['input_ids'][0].unsqueeze(0))
#print(outputs)
#print(len(outputs['hidden_states']))

trainer = CustomTrainer(
        model,
        args,
        train_dataset = encoded_dataset["train"],
        eval_dataset  = encoded_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=partial(common.compute_metrics, labels=labels)
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
trainer.save_model(f"best_{num_train_epochs}_{batch_size}")
#common.show_memory("Memory after Evaluation")
common.save_eval_result_df = None
print(results)
cmd = subprocess.run(["python", "evaluator.py",
                      "--inputDataset", "evaluationLabels",
                      "--inputRun", "evaluationResults",
                      "--outputDataset", "evaluationResults"])
