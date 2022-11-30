import torch

import sys
sys.path.append('../Utils/')

from bert import *
from loader import *
from settings import *
from ml_models import *

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

def train(epoch, tloader, device, optimizer, model):
    model.train()
    
    for _,data in enumerate(tloader, 0):
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.float)

        outputs = model(ids, mask, token_type_ids)

        loss = loss_fn(outputs, targets)
        if _%500 == 0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dl = Loader()
    dl.load()
    dl.stem()
    # dl.tknz()
    run_logistic_regression(dl)

    # dl.split_to_train_val()
    # tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    # model = BERTClass(target_cols=dl.get_target_cols())
    # model.to(device);
    # optimizer = AdamW(params =  model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)

    # train_dataset = BERTDataset(dl.train, tokenizer, MAX_LEN, target_cols=dl.get_target_cols())
    # valid_dataset = BERTDataset(dl.validation, tokenizer, MAX_LEN, target_cols=dl.get_target_cols())
    # train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, num_workers=4, shuffle=True, pin_memory=True)
    # valid_loader = DataLoader(valid_dataset, batch_size=VALID_BATCH_SIZE, num_workers=4, shuffle=False, pin_memory=True)

    # for epoch in range(EPOCHS):
    #     train(epoch, train_loader, device, optimizer, model)


if __name__ == '__main__':
    main()