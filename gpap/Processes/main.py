import sys
sys.path.append('../Utils/')

from bert import *
from many_single_class_BERT_models import *
from gan_bert import *
from loader import *
from settings import *
from ml_models import *
from sklearn.svm import SVC

import transformers
transformers.logging.set_verbosity_error()

import torch

def main():

    # Define data loader(s)
    if not W_NEW_DATA:
        dl = Loader()
        dl.load(clean=False, w_sep=True, w_concl=True, w_stance=W_STANCE)
    else:
        train_dl = Loader(data_base_path='../new_Data/',
                          ta_file_name='arguments-training.tsv',
                          tl_file_name='labels-training.tsv',
                          workingTableType='value_categories',
                          unlabeled_data_file_name='arguments-test.tsv',
                          with_unlabeled_data=W_UNLABELED_DATA)

        val_dl = Loader(data_base_path='../new_Data/',
                        ta_file_name='arguments-validation.tsv',
                        tl_file_name='labels-validation.tsv',
                        workingTableType='value_categories')
        test_dl = Loader(data_base_path='../new_Data/',
                        ta_file_name='arguments-test.tsv',
                        tl_file_name=None,
                        workingTableType='value_categories')

        train_dl.load(clean=False, w_sep=True, w_concl=True, w_stance=W_STANCE)
        val_dl.load(clean=False, w_sep=True, w_concl=True, w_stance=W_STANCE)
        test_dl.load(clean=False, w_sep=True, w_concl=True, w_stance=W_STANCE)

    # Define algorithm
    if ALGO=='SVM':

        dl.stem()
        dl.tknz()

        run_multioutput_clf(dl, clf=SVC())

    elif ALGO == 'BERT':

        if not W_NEW_DATA:
            dl.StratifiedCFV(n_folds=1)
            max_length = dl.get_max_len()[0] if MAX_LENGTH == -1 else MAX_LENGTH
            target_cols = dl.get_target_cols()
        else:
            max_length = train_dl.get_max_len()[0] if MAX_LENGTH == -1 else MAX_LENGTH
            target_cols = train_dl.get_target_cols()

        # Check the consistency of settings
        if TEST_ONLY and GAN_BERT:
            print('Not implemented !')
            exit(0)
        if TEST_ONLY and not W_NEW_DATA:
            print('TEST_ONLY and W_NEW_DATA should be both True because old data do not contain a test set !!!')
            exit(0)
        if TRAIN and TEST_ONLY:
            print('TRAIN and TEST_ONLY cannot be both True !!!')
            exit(0)
        if EVALUATE_ONLY and TEST_ONLY:
            print('EVALUATE_ONLY and TEST_ONLY cannot be both True !!!')
            exit(0)
        if TRAIN and EVALUATE_ONLY:
            print('EVALUATE_ONLY and TRAIN cannot be both True !!!')
            exit(0)
        if GAN_BERT and EVALUATION_W_MANY_SINGLE_CLASS_BERT_MODELS:
            print('GAN_BERT and EVALUATION_W_MANY_SINGLE_CLASS_BERT_MODELS cannot be both True !!!')
            exit(0)
        if (GAN_BERT and not ONLY_BERT_EMBEDDINGS) or (not GAN_BERT and ONLY_BERT_EMBEDDINGS):
            print('GAN_BERT and ONLY_BERT_EMBEDDINGS should be both True or both False !!!')
            exit(0)
        if (not GAN_BERT and W_UNLABELED_DATA):
            print('W_UNLABELED_DATA cannot be True while GAN_BERT is False !!!')
            exit(0)
        if (not W_NEW_DATA and W_UNLABELED_DATA):
            print('W_UNLABELED_DATA cannot be True while W_NEW_DATA is False !!!')
            exit(0)
        if EVALUATION_W_MANY_SINGLE_CLASS_BERT_MODELS and TRAIN:
            print('EVALUATION_W_MANY_SINGLE_CLASS_BERT_MODELS and TRAIN cannot be both true !!!')
            exit(0)

        if GAN_BERT:
            model = GanBERT(target_cols=target_cols, max_length=max_length, dl=(None if W_NEW_DATA else dl),
                            loader_train_dataset=(None if not W_NEW_DATA else train_dl),
                            loader_valid_dataset=(None if not W_NEW_DATA else val_dl),
                            device='cuda' if torch.cuda.is_available() else 'cpu',
                            discr_hidden_size=512, num_labels=len(target_cols), gen_noise_size=100, gen_hidden_size=512,
                            epsilon=1e-8)

        elif EVALUATION_W_MANY_SINGLE_CLASS_BERT_MODELS:
            model = ManySingleClassBertModels(target_cols, max_length,
                                              dl=(None if W_NEW_DATA else dl),
                                              loader_train_dataset=(None if not W_NEW_DATA else train_dl),
                                              loader_valid_dataset=(None if not W_NEW_DATA else val_dl),
                                              loader_test_dataset=(None if not W_NEW_DATA else test_dl))

        else:
            model = BERTClass(target_cols=target_cols, max_length=max_length,
                              dl=(None if W_NEW_DATA else dl),
                              loader_train_dataset=(None if not W_NEW_DATA else train_dl),
                              loader_valid_dataset=(None if not W_NEW_DATA else val_dl),
                              loader_test_dataset=(None if not W_NEW_DATA else test_dl),
                              device='cuda' if torch.cuda.is_available() else 'cpu')

        if TRAIN:
            model.train_()
        elif EVALUATE_ONLY:
            model.evaluate_()
        elif TEST_ONLY:
            model.test_()

    else:
        print("You should choose a valid algorithm !!!")
        exit(0)


if __name__ == '__main__':
    main()