import sys
sys.path.append('../Utils/')

from bert import *
from loader import *
from settings import *
from ml_models import *
from sklearn.svm import SVC

import transformers
transformers.logging.set_verbosity_error()

def main():

    dl = Loader()
    dl.load(clean=False, w_sep=True, w_concl=False, w_stance=False)

    if ALGO=='SVM':

        dl.stem()
        dl.tknz()

        run_multioutput_clf(dl, clf=SVC())

    elif ALGO == 'BERT':

        dl.train_test_validate_split(test_size=0.3, to_loader=True, random_state=1)
        max_length = dl.get_max_len()[0]

        model = BERTClass(target_cols=dl.get_target_cols(), max_length=max_length, freeze_bert=True, head_type='GRU', multihead=True, dl=dl)
        model.train_()

    else:
        print("You should choose a valid algorithm !!!")
        exit(0)


if __name__ == '__main__':
    main()