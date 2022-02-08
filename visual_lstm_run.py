import time
import torch
import pickle
import numpy as np
import random as rn
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import DataLoader

from models import VisualConcatLSTMModel
from data import VisualLSTMDataset

seed = 0
rn.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# load the data
glove_filename = '../glove.840B.300d.txt'
dev_filename = '../vsnli/VSNLI_1.0_dev.tsv'
train_filename = '../vsnli/VSNLI_1.0_train.tsv'
test_filename = '../vsnli/VSNLI_1.0_test.tsv'
test_hard_filename = '../vsnli/VSNLI_1.0_test_hard.tsv'

max_vocab = 300000
embedding_size = 300
hidden_size = 512
batch_size = 8192
epochs = 10
lr = 0.001
modelname = 'vgg16_bn'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('metadata' + ".index", mode="rb") as out_file:
    metadata = pickle.load(out_file)

token2id = metadata["token2id"] 
id2token = metadata["id2token"] 
label2id = metadata["label2id"] 
id2label = metadata["id2label"]

embeddings = torch.tensor(metadata["embeddings"], dtype=torch.float32, requires_grad=True)
num_labels = len(label2id)  

with open('../data/vte/' + "dev.pkl", mode="rb") as out_file: dev = pickle.load(out_file)
with open('../data/vte/' + "train.pkl", mode="rb") as out_file: train = pickle.load(out_file)
with open('../data/vte/' + "test.pkl", mode="rb") as out_file: test = pickle.load(out_file)
with open('../data/vte/' + "test_hard.pkl", mode="rb") as out_file: test_hard = pickle.load(out_file)

print("\n-------------- DATA STATISTICS --------------")
print(f"Number of labels: {num_labels}")     
print(f"Dev size: {len(dev['labels'])}")
print(f"Train size: {len(train['labels'])}")
print(f"Test size: {len(test['labels'])}")
print(f"Test_hard size: {len(test_hard['labels'])}")
print("-------------- DATA STATISTICS --------------\n")

with open(f'../data/vte/{modelname}/' + "dev.npy", mode="rb") as out_file: dev_features = np.load(out_file)
with open(f'../data/vte/{modelname}/' + "test.npy", mode="rb") as out_file: test_features = np.load(out_file)
with open(f'../data/vte/{modelname}/' + "test_hard.npy", mode="rb") as out_file: test_hard_features = np.load(out_file)
# with open(f'../data/vte/{modelname}/' + "train.npy", mode="rb") as out_file: train_features = np.load(out_file)

train_dataset = VisualLSTMDataset(test['padded_premises'], test['padded_hypotheses'], test['labels'], 
                                    test['image_name'], test_features)
dev_dataset = VisualLSTMDataset(dev['padded_premises'], dev['padded_hypotheses'], dev['labels'],
                                    dev['image_names'], dev_features)

train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=train_dataset.collater,
            shuffle=True,
        )
dev_dataloader = DataLoader(
            dev_dataset,
            batch_size=batch_size,
            collate_fn=dev_dataset.collater,
            shuffle=False,
        )

model = VisualConcatLSTMModel(embeddings, embedding_size, hidden_size, num_labels)

if torch.cuda.device_count() > 0:
    model = nn.DataParallel(model)

model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    total_loss = 0
    for i, batch in enumerate(train_dataloader):
        labels = batch['labels'].to(device)
        premises = batch['premises'].to(device)
        hypotheses = batch['hypotheses'].to(device)
        features = batch['features'].to(device)

        out = model(premises, hypotheses, features)

        loss = criterion(out, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 

        total_loss += loss.item()
        
        if (i+1) % 1 == 0:
            print(f"({i+1} batches completed) Loss: {total_loss/(i+1):.4f}")

    with torch.no_grad():
        total_loss = 0
        correct, total = 0, 0

        for i, batch in enumerate(dev_dataloader):
            labels = batch['labels'].to(device)
            premises = batch['premises'].to(device)
            hypotheses = batch['hypotheses'].to(device)
            features = batch['features'].to(device)

            out = model(premises, hypotheses, features)

            loss = criterion(out, labels)
            total_loss += loss.item()

            out = torch.nn.functional.softmax(out, dim=1)
            _, predicted = torch.max(out, dim=1)
            correct += (predicted == labels).sum().item()
            total += len(labels)

        print(f"Dev loss: {total_loss/len(dev_dataloader):.4f} Examples: {total}")
        print(f"Dev accuracy: {correct/total * 100}")