TRAIN_BATCH_SIZE = 8
EPOCHS = 1000
LEARNING_RATE = 2e-5
PATIENCE = 20
DROPOUT = 0.5
HIDDEN_DROPOUT_PROB = 0.1
ATTENTION_PROBS_DROPOUT_PROBS = 0.2
ALGO = 'BERT'
FREEZE_BERT = False
ONLY_BERT_EMBEDDINGS = False
GAN_BERT = False
FAKE_OR_REAL_PROBS_METHOD = "Softmax" # "Softmax" or "BCE"
W_UNLABELED_DATA = False #Works with GAN-BERT
UPSAMPLING_RATE_UNLABELED_DATA = 2
HEAD_TYPE = 'MLP' #Options: 'MLP', 'GRU'
MULTIHEAD = False
BIODIRECTIONAL_GRU = True
GRU_HIDDEN_DIM = 20
MODEL_PATH = './../saved_model/model.pt'
MODEL_PATH_FOR_EVALUATION_ONLY = './../saved_model/model.pt'
EVALUATION_W_MANY_SINGLE_CLASS_BERT_MODELS = False
MAX_LENGTH = 128 #If -1 it will be specified automatically based on the length of the sentence with the max number of words
LOSS = 'BCE' # 'BCE', 'sigmoidF1'
SINGLE_CLASS = False
CLASS = 0
W_LOSS_WEIGHTS = False
TRAIN = False
EVALUATE_ONLY = False
TEST_ONLY = True
W_CLS_ONLY_FIX = True
W_NEW_DATA = True
W_MASKING = False
MAKING_PERC = 15/100
ADD_SPECIAL_TOKENS = False
W_STANCE = False
DISCR_DROPOUT_RATE = 0.1
DISCR_LR = 2e-5
GEN_DROPOUT_RATE = 0.1
GEN_LR = 2e-5