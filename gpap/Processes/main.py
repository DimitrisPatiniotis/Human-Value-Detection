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

    dl = Loader()
    dl.load(clean=False, w_sep=True, w_concl=True, w_stance=False)

    if ALGO=='SVM':

        dl.stem()
        dl.tknz()

        run_multioutput_clf(dl, clf=SVC())

    elif ALGO == 'BERT':

        dl.StratifiedCFV(n_folds=1)
        max_length = dl.get_max_len()[0] if MAX_LENGTH == -1 else MAX_LENGTH

        model = BERTClass(target_cols=dl.get_target_cols(), max_length=max_length, dl=dl,
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