import sys
sys.path.append('../Utils/')

from bert import *
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
        dl.load(clean=False, w_sep=True, w_concl=True, w_stance=False)
    else:
        train_dl = Loader(data_base_path='../new_Data/',
                          ta_file_name='arguments-training.tsv',
                          tl_file_name='labels-training.tsv',
                          workingTableType='value_categories')

        val_dl = Loader(data_base_path='../new_Data/',
                        ta_file_name='arguments-validation.tsv',
                        tl_file_name='labels-validation.tsv',
                        workingTableType='value_categories')

        train_dl.load(clean=False, w_sep=True, w_concl=True, w_stance=False)
        val_dl.load(clean=False, w_sep=True, w_concl=True, w_stance=False)

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

        model = BERTClass(target_cols=target_cols, max_length=max_length,
                          dl=(None if W_NEW_DATA else dl),
                          loader_train_dataset=(None if not W_NEW_DATA else train_dl),
                          loader_valid_dataset=(None if not W_NEW_DATA else val_dl),
                          device='cuda' if torch.cuda.is_available() else 'cpu')

        if TRAIN:
            model.train_()
        else:
            model.evaluate_()

    else:
        print("You should choose a valid algorithm !!!")
        exit(0)


if __name__ == '__main__':
    main()