TRAIN_BATCH_SIZE = 8
EPOCHS = 1000
LEARNING_RATE = 2e-5
PATIENCE = 20
DROPOUT = 0.5
HIDDEN_DROPOUT_PROB = 0.1
ATTENTION_PROBS_DROPOUT_PROBS = 0.1
ALGO = 'BERT'
FREEZE_BERT = False
HEAD_TYPE = 'MLP' #Options: 'MLP', 'GRU'
MULTIHEAD = False
BIODIRECTIONAL_GRU = True
GRU_HIDDEN_DIM = 20
MODEL_PATH = './../saved_model/model.pt'
MAX_LENGTH = 128 #If -1 it will be specified automatically based on the length of the sentence with the max number of words
LOSS = 'sigmoidF1' # 'BCE', 'sigmoidF1'
SINGLE_CLASS_TRAINING = True
CLASS = 0
W_LOSS_WEIGHTS = True