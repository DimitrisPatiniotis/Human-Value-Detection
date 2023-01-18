from common import common
from transformers import AutoTokenizer
from transformers import AutoModel, AutoModelForSequenceClassification, AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer
from common.trainers import CustomTrainer
from functools import partial
import subprocess
import numpy as np
from collections import defaultdict, Counter
from datasets import Dataset
from common.kmeans import KMeans
import torch
import os.path

use_cuda = torch.cuda.is_available()
dtype = torch.float32 if use_cuda else torch.float64
device_id = "cuda:0" if use_cuda else "cpu"

common.setSeeds(2022)

## Read dataset...
datadir = '../Data'
df_train, df_validation, df_test = common.getData(datadir)
dataset = common.getDatasets(df_train, df_validation, df_test)

## Get dataset labels...
labels = [label for label in dataset['train'].features.keys() if label not in common.dataLabels]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}
# print("Labels:", labels)

############################################################
## Parameters
############################################################
pretrained_model_name = "./best_130_4"
pretrained_model_name = "bert-base-uncased"
batch_size            = 8
metric_name           = "f1"
num_train_epochs      = 30
freeze_layers_bert    = False
number_of_centroids   = 4
calculate_centroids   = False
centroids_fname       = 'centroids.pt'
if not os.path.exists(centroids_fname):
    centroids_fname = True

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
                pretrained_model_name,
                problem_type="multi_label_classification",
                output_hidden_states=True,
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

trainer = CustomTrainer(
        model,
        args,
        train_dataset = encoded_dataset["train"],
        eval_dataset  = encoded_dataset["validation"],
        tokenizer=tokenizer,
        compute_metrics=partial(common.compute_metrics, labels=labels)
)

## Forward pass
if calculate_centroids:
    print("Calculating Centroids!")
    trainer.tokenid2embeddings = defaultdict(list)
    for part in ["test", "validation", "train"]:
        # for example in encoded_dataset[part]:
        #     print(example)
        #     print(part, tokenizer.decode(example['input_ids']))
        #     break

        outputs = trainer.predict(test_dataset=encoded_dataset[part])
        #break

        # for example in encoded_dataset[part]:
        #     print(part, tokenizer.decode(example['input_ids']))
        #     #outputs = trainer.predict(test_dataset=Dataset.from_dict(example))
        #     last_hidden_state = outputs.hidden_states[-1][0]
        #     for token_index, token in enumerate(example['input_ids']):
        #         if token < 1000:
        #             continue
        #         id2embeddings[token].append(last_hidden_state[token_index].detach().numpy())
    counter = Counter()
    for key in trainer.tokenid2embeddings:
        token = tokenizer.decode([key])
        l  = len(trainer.tokenid2embeddings[key])
        if l > 1:
            counter[key] += l

    tokenid2centroids = {}
    for key, freq in counter.most_common():

        ## Perform kmeans clustering...
        x = torch.from_numpy(np.array(trainer.tokenid2embeddings[key]))
        print(key, '->', freq, x.shape)
        # print(trainer.tokenid2embeddings[key])
        if freq < number_of_centroids:
            tokenid2centroids[key] = x
        else:
            cl, c = KMeans(x, number_of_centroids)
            tokenid2centroids[key] = c
        #print(cl, cl.shape)
        #print(c, c.shape)
    torch.save(tokenid2centroids, centroids_fname)
    trainer.tokenid2embeddings = None
else:
    print(f"Loading Centroids from {centroids_fname}!")
    tokenid2centroids = torch.load(centroids_fname)
trainer.centroids(tokenid2centroids)

# outputs = model(input_ids=encoded_dataset['validation']['input_ids'][0].unsqueeze(0))
# print(outputs)
# print(len(outputs['hidden_states']))

#common.show_memory("Memory before Training")
print("############### Training:")
trainer.train()
#common.show_memory("Memory after Training")
print("############### Evaluation:")
#common.save_eval_result_df = df_validation
results = trainer.evaluate()
#trainer.save_model(f"best_{num_train_epochs}_{batch_size}")
#common.show_memory("Memory after Evaluation")
common.save_eval_result_df = None
print(results)
exit(0)
cmd = subprocess.run(["python", "evaluator.py",
                      "--inputDataset", "evaluationLabels",
                      "--inputRun", "evaluationResults",
                      "--outputDataset", "evaluationResults"])
