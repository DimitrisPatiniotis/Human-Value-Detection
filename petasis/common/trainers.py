from transformers import Trainer
from transformers.modeling_outputs import SequenceClassifierOutput
import numpy as np

from collections import defaultdict

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs     = super().compute_loss(model=model, inputs=inputs, return_outputs=True)
        #  input_ids         = inputs.get("input_ids").cpu()
        #  last_hidden_state = outputs.hidden_states[-1]
        #  id2embeddings = defaultdict(list)
        #  for example_index, example in enumerate(input_ids):
        #      example_last_hidden_state = last_hidden_state[example_index]
        #      print(example_last_hidden_state.shape, example.shape)
        #      for token_index, token in enumerate(example):
        #          if token < 1000:
        #              continue
        #          id2embeddings[token].append(example_last_hidden_state[token_index])
        #          #print(example_index, token_index, token)
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
            ## Remove hidden_states, or we will exhaust VRAM.
            outputs = SequenceClassifierOutput(
                loss=outputs.loss,
                logits=outputs.logits,
                hidden_states=outputs.hidden_states[-1].detach().cpu(),
                attentions=outputs.attentions,
            )
        return (loss, outputs) if return_outputs else loss
