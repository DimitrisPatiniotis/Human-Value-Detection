import common
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from functools import partial
import subprocess

common.setSeeds(2022)

## Read dataset...
datadir = '../Data'
df_train, df_validation, df_test = common.getData(datadir)
dataset = common.getDatasets(df_train, df_validation, df_test)

## Get dataset labels...
labels = [label for label in dataset['train'].features.keys() if label not in ['Argument ID', 'Conclusion', 'Stance', 'Premise', '__index_level_0__']]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}
# print("Labels:", labels)

############################################################
## Parameters
############################################################
pretrained_model_name = "bert-base-uncased"
pretrained_model_name = "distilbert-base-uncased"
batch_size            = 8
metric_name           = "f1"
num_train_epochs      = 15
freeze_layers_bert    = False

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
encoded_dataset = common.encodeDataset(dataset, labels, tokenizer)

def instantiate_model(pretrained_model_name, freezeLayers=False):
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name, 
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
trainer = Trainer(
        model,
        args,
        train_dataset = encoded_dataset["train"],
        eval_dataset  = encoded_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=partial(common.compute_metrics, labels=labels)
)
print("############### Training:")
trainer.train()
print("############### Evaluation:")
common.save_eval_result_df = df_validation
results = trainer.evaluate()
common.save_eval_result_df = None
print(results)
cmd = subprocess.run(["python", "evaluator.py",
                      "--inputDataset", "evaluationLabels",
                      "--inputRun", "evaluationResults",
                      "--outputDataset", "evaluationResults"])
