import sys
sys.path.append('../Utils/')
from bert import *
from settings import *

import torch
import transformers
transformers.logging.set_verbosity_error()

def many_to_one(num_models=20, path_to_models='./'):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Define the single-head models
    single_head_models_list = \
        [BERTClass(target_cols=['dummy_col'], max_length=MAX_LENGTH,
                   dl=None, device=device, multihead=False)
         for model_i in range(num_models)]

    # Load the stored weights of the single-head models
    for model_i in range(num_models):
        single_head_models_list[model_i].\
            load_state_dict(torch.load(path_to_models + '/saved_model_class_' + str(model_i) + '/model.pt',
                                       map_location=torch.device(device)))

    # Keep only the parameters (weights and biases) of the last layer (classification head) of each single-head model
    single_head_models_last_layer_weights_bias_list = \
        [{'weights': model_i.fc.weight.detach().numpy(),
          'bias': model_i.fc.bias.detach().numpy()}
         for model_i in single_head_models_list]

    # Define the multi-head model
    multi_head_model = BERTClass(target_cols=['dummy_col_'+str(class_i) for class_i in range(num_models)],
                                 max_length=MAX_LENGTH, dl=None, device=device, multihead=True)

    # Assign the previously kept parameters to the multi-head layer
    for head_i in range(num_models):

        multi_head_model.fcs[head_i].weight = \
            torch.nn.Parameter(torch.from_numpy(single_head_models_last_layer_weights_bias_list[head_i]['weights']).float())

        multi_head_model.fcs[head_i].bias = \
            torch.nn.Parameter(torch.from_numpy(single_head_models_last_layer_weights_bias_list[head_i]['bias']).float())

    # Save the multi-head model
    torch.save(multi_head_model.state_dict(), path_to_models + '/saved_model_w_multi_heads/model.pt')


if __name__ == '__main__':

    many_to_one(num_models=20,
                path_to_models='./../../saved_model_trained_one_by_one')


