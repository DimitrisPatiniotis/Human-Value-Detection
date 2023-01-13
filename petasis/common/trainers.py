from transformers import Trainer
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, CosineEmbeddingLoss
import numpy as np
import torch
from pykeops.torch import LazyTensor

from collections import defaultdict, Counter
from .kmeans import knn
import math

class CustomTrainer(Trainer):
    tokenid2embeddings = None
    tokenid2centroids  = None
    use_class_weights: bool = False
    class_weights      = None

    def compute_loss(self, model, inputs, return_outputs=False):
        torch.autograd.set_detect_anomaly(True)
        # loss has been caluclated as BCEWithLogitsLoss(logits, labels)
        loss, outputs     = super().compute_loss(model=model, inputs=inputs, return_outputs=True)
        #print("LOSS:", loss, outputs.logits.shape, self.is_model_parallel )
        #print(outputs)
        if self.use_class_weights:
            labels = inputs.get("labels")
            loss_fct = BCEWithLogitsLoss(pos_weight=self.class_weights.to(loss.device))
            loss2 = loss_fct(outputs.logits, labels)
            # print("Loss:", loss, "->", loss2)
            loss = loss2
        if self.tokenid2embeddings is not None:
            input_ids         = inputs.get("input_ids").cpu()
            last_hidden_state = outputs.hidden_states[-1]
            for example_index, example in enumerate(input_ids):
                example_last_hidden_state = last_hidden_state[example_index]
                # print(example_last_hidden_state.shape, example.shape)
                for token_index, token in enumerate(example):
                    token = token.item()
                    if token < 1000:
                        continue
                    self.tokenid2embeddings[token].append(example_last_hidden_state[token_index].cpu().tolist())
                    #print(example_index, token_index, token)
        if self.tokenid2centroids is not None:
            input_ids         = inputs.get("input_ids").cpu()
            last_hidden_state = outputs.get("hidden_states")[-1]
            t_loss = None
            for example_index, example in enumerate(input_ids):
                example_last_hidden_state = last_hidden_state[example_index]
                for token_index, token in enumerate(example):
                    token = token.item()
                    if token < 1000:
                        continue
                    if token in self.tokenid2centroids:
                        # print("Found:", token)
                        x = example_last_hidden_state[token_index]
                        dm = None
                        for c in self.tokenid2centroids[token].to(x.device):
                            # d = torch.sqrt(((x - c) ** 2).sum())
                            d = ((x - c) ** 2).sum()
                            if math.isnan(d):
                                # print("NAN:", token, dm)
                                # print("d:", token, d)
                                # print("x:", token, x)
                                # print("c:", token, c)
                                continue
                            if dm is None:
                                dm = d
                            else:
                                if d < dm:
                                    dm = d
                            #print(dm, d)
                        # print(token, dm)
                        if t_loss is not None:
                            t_loss += dm
                        else:
                            t_loss = dm
                        # x = torch.unsqueeze(example_last_hidden_state[token_index], 0) # 1 * 768
                        # x = example_last_hidden_state[token_index]
                        # c = self.tokenid2centroids[token].float().to(x.device)
                        # print("X:", x, x.shape)
                        # print("C:", c, c.shape)
                        # x_i = LazyTensor( x[None, None, :] ) # (N, 1, D) samples
                        # c_j = LazyTensor( c[None,:,:] ) # (1, K, D) centroids
                        # print("x_i", x_i)
                        # print("c_j", c_j)
                        # D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
                        # print("D_ij:", D_ij)
                        # cl = D_ij.argmin(1, dim=1)  # Points -> Nearest cluster
                        # print("cl:" , cl)
            loss += t_loss
            print("loss:", loss)
        #  for key in id2embeddings:
        #      print(key, len(id2embeddings[key]))
        # unique_input_ids = np.unique(input_ids)
        # print(outputs.keys())
        # labels        = inputs.get("labels")
        # logits        = outputs.get("logits")
        # hidden_states = outputs.get("hidden_states")
        # compute custom loss
        # print(inputs)
        # print("compute_loss:", loss, hidden_states[12][0], len(hidden_states), len(hidden_states[12][0]), return_outputs)
        # exit(0)
        if return_outputs:
            ## Move hidden_states to cpu(), or we will exhaust GPU RAM.
            outputs = SequenceClassifierOutput(
                loss=outputs.loss,
                logits=outputs.logits,
                #hidden_states=outputs.hidden_states[-1].cpu(),
                attentions=outputs.attentions,
            )
        return (loss, outputs) if return_outputs else loss

    def centroids(self, centroids=None):
        self.tokenid2centroids = centroids

    def class_weights(self, weights=None):
        if weights is None:
            self.use_class_weights = False
            self.class_weights     = None
        else:
            self.use_class_weights = True
            self.class_weights     = torch.from_numpy(weights)
