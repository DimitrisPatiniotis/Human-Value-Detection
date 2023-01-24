from bert import *
from loader import *
from settings import *
from losses import *
from dataset_constructor import BERTDataset

from tqdm import tqdm
import torch
import transformers
transformers.logging.set_verbosity_error()

class ManySingleClassBertModels(torch.nn.Module):
    def __init__(self, target_cols, max_length, dl=None,
                 loader_train_dataset=None, loader_valid_dataset=None,
                 loader_test_dataset=None):

        super(ManySingleClassBertModels, self).__init__()

        self.loader_train_dataset = loader_train_dataset
        self.loader_valid_dataset = loader_valid_dataset
        self.loader_test_dataset = loader_test_dataset
        self.dl = dl
        self.max_length = max_length
        self.target_cols = target_cols
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

        self.models_list = \
            torch.nn.ModuleList([BERTClass(target_cols=[self.target_cols[i]], max_length=max_length,
                                           dl=(None if W_NEW_DATA else self.dl),
                                           loader_train_dataset=(None if not W_NEW_DATA else self.loader_train_dataset),
                                           loader_valid_dataset=(None if not W_NEW_DATA else self.loader_valid_dataset),
                                           loader_test_dataset=(None if not W_NEW_DATA else self.loader_test_dataset),
                                           device='cuda' if torch.cuda.is_available() else 'cpu')
                                 for i in range(len(self.target_cols))])

    def forward(self, ids):

        # ids = [batch size, sent len]

        outputs = [BERT_model(ids) for BERT_model in self.models_list]
        # outputs = list of 20 tensors each of size [batch size, 1]

        output = torch.concat(outputs, dim=1)
        # output = [batch size, num classes]

        return output

    def best_model_evaluation(self, vloader, device, evaluation_only=False, write_file=False, path_to_models='./'):

        if not evaluation_only:
            print('Training has ended. Loading best model for evaluation..')
        else:
            print('Loading saved models for evaluation..')

        # Load the stored weights of the single-head models
        for model_i in tqdm(range(len(self.target_cols))):
            self.models_list[model_i].\
                load_state_dict(torch.load(path_to_models + '/saved_model_class_' + str(model_i) + '/model.pt',
                                           map_location=torch.device(device)))

        # Turn off train mode
        self.eval()

        final_v_loss_list = []
        final_v_pred_one_hot = []
        final_v_ground_truth = []
        for _, v_data in tqdm(enumerate(vloader, 0)):
            v_ids = v_data['ids'].to(device, dtype=torch.long)
            v_targets = v_data['targets'].to(device, dtype=torch.float)

            with torch.no_grad():
                v_outputs = self(v_ids)

                if W_LOSS_WEIGHTS:
                    v_loss_weights_repeated_batch_size = \
                        np.repeat(self.train_dataset.train_loss_weights[np.newaxis, :], v_outputs.size()[0], axis=0)

                    v_final_loss_weights = \
                        np.take_along_axis(v_loss_weights_repeated_batch_size,
                                           np.array(v_targets.cpu().numpy() if v_targets.cpu().numpy().shape[1] == 1
                                                                            else
                                                    np.expand_dims(v_targets.cpu().numpy(), axis=2),
                                                    dtype=int),
                                           axis=1 if v_targets.cpu().numpy().shape[1] == 1 else 2)

                    if v_targets.cpu().numpy().shape[1] > 1:
                        v_final_loss_weights = np.squeeze(v_final_loss_weights, axis=2)

                v_loss = BCE_loss(v_outputs, v_targets,
                                  None if not W_LOSS_WEIGHTS
                                       else torch.from_numpy(v_final_loss_weights).to(device, dtype=torch.float32)) \
                         if LOSS == 'BCE' \
                         else (f1_loss(v_outputs, v_targets,
                                       None if not W_LOSS_WEIGHTS
                                            else torch.from_numpy(v_final_loss_weights).to(device, dtype=torch.float32))
                               if LOSS == 'sigmoidF1'
                               else 0.0)

                final_v_loss_list.append(v_loss if LOSS == 'sigmoidF1' else v_loss.item())

                sig = torch.nn.Sigmoid()
                v_probs = sig(v_outputs)

            final_v_pred_one_hot.append(np.around(v_probs.cpu().numpy()))
            final_v_ground_truth.append(v_targets.cpu().numpy())

        v_conc_pred_one_hot = np.concatenate(final_v_pred_one_hot, axis=0)
        v_conc_ground_truth = np.concatenate(final_v_ground_truth, axis=0)
        final_f1_micro_average = f1_score(y_true=v_conc_ground_truth,
                                          y_pred=v_conc_pred_one_hot,
                                          average='micro',
                                          zero_division=0)
        v_clr_dict = classification_report(v_conc_ground_truth,
                                           v_conc_pred_one_hot,
                                           zero_division=0)
        final_f, final_pr, final_rec, final_fs, final_prs, final_recs = \
            F1_as_evaluator(v_conc_pred_one_hot, v_conc_ground_truth, self.target_cols)

        print(f'Best model: \n'
              f'Val Loss: {sum(final_v_loss_list) / len(final_v_loss_list):.2f}, '
              f'Val micro F1: {final_f1_micro_average:.2f}\n'
              f'Classification Report: \n {v_clr_dict}\n'
              f' Val evaluator F1: {final_f:.2f}, '
              f'Val evaluator Precision: {final_pr:.2f}, '
              f'Val evaluator Recall: {final_rec:.2f}\n')

        for value_col_id, value_col in enumerate(self.target_cols):
            print("measure {\n key: \"Precision " + value_col + "\"\n value: \"" + str(final_prs[value_col_id]) + "\"\n}\n" +
                  "measure {\n key: \"Recall " + value_col + "\"\n value: \"" + str(final_recs[value_col_id]) + "\"\n}\n" +
                  "measure {\n key: \"F1 " + value_col + "\"\n value: \"" + str(final_fs[value_col_id]) + "\"\n}\n")

        if write_file:
            writeRun(labels=pd.DataFrame(v_conc_pred_one_hot, columns=self.target_cols),
                     argument_ids=self.loader_valid_dataset.workingTable["Argument ID"].tolist(),
                     outputDataset='./')
            evaluateRun(self.loader_valid_dataset.DATA_PATH + 'validation_labels_only', './', './')

    def evaluate_(self, path_to_models='./../../..'):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        if self.max_length > 512:
            tokenizer.model_max_length = self.max_length
            print('Max len exceeds 512 tokens !!! Tokenizer is changed !!!')

        self.to(device)

        if not W_NEW_DATA:
            validation_df_for_BERTDataset = self.dl.validation
        else:
            validation_df_for_BERTDataset = self.loader_valid_dataset.workingTable

        self.valid_dataset = BERTDataset(validation_df_for_BERTDataset, tokenizer,
                                         self.max_length, target_cols=self.target_cols)
        valid_loader = DataLoader(self.valid_dataset, batch_size=TRAIN_BATCH_SIZE,
                                  num_workers=4, shuffle=False, pin_memory=True)

        self.best_model_evaluation(valid_loader, device, evaluation_only=True,
                                   path_to_models=path_to_models, write_file=True)

    def test_(self, path_to_models='./../../..'):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        if self.max_length > 512:
            tokenizer.model_max_length = self.max_length
            print('Max len exceeds 512 tokens !!! Tokenizer is changed !!!')

        self.to(device)

        test_df_for_BERTDataset = self.loader_test_dataset.workingTable
        self.test_dataset = BERTDataset(test_df_for_BERTDataset, tokenizer,
                                        self.max_length, target_cols=self.target_cols,
                                        test=True)
        test_loader = DataLoader(self.test_dataset, batch_size=TRAIN_BATCH_SIZE,
                                  num_workers=4, shuffle=False, pin_memory=True)

        print('Loading saved models for test..')
        # Load the stored weights of the single-head models
        for model_i in tqdm(range(len(self.target_cols))):
            self.models_list[model_i]. \
                load_state_dict(torch.load(path_to_models + '/saved_model_class_' + str(model_i) + '/model.pt',
                                           map_location=torch.device(device)))

        # Turn off train mode
        self.eval()

        t_pred_one_hot = []
        for _, t_data in tqdm(enumerate(test_loader, 0)):
            t_ids = t_data['ids'].to(self.device, dtype=torch.long)

            with torch.no_grad():
                t_outputs = self(t_ids)

                sig = torch.nn.Sigmoid()
                t_probs = sig(t_outputs)

            t_pred_one_hot.append(np.around(t_probs.cpu().numpy()))

        t_conc_pred_one_hot = np.concatenate(t_pred_one_hot, axis=0)

        writeRun(labels=pd.DataFrame(t_conc_pred_one_hot, columns=self.target_cols),
                 argument_ids=self.loader_test_dataset.workingTable["Argument ID"].tolist(),
                 outputDataset='./')


