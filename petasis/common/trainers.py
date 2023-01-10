from transformers import Trainer
import numpy as np

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        loss, outputs    = super().compute_loss(model=model, inputs=inputs, return_outputs=True)
        input_ids        = inputs.get("input_ids").cpu()
        unique_input_ids = np.unique(input_ids)
        # labels        = inputs.get("labels")
        # logits        = outputs.get("logits")
        # hidden_states = outputs.get("hidden_states")
        # compute custom loss
        # print(inputs)
        # print("compute_loss:", loss, len(hidden_states), len(hidden_states[12][0]))
        return (loss, outputs) if return_outputs else loss
