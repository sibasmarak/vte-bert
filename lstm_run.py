import torch
import time,os
import pickle
import numpy as np
import random as rn
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# NOTE: Uncomment the following for creating the computation graph
# from torchviz import make_dot, make_dot_from_trace
# import matplotlib.pyplot as plt
# from graphviz import Source

from data import LSTMDataset, VisualLSTMDataset
from models import VanillaLSTMModel, ConcatLSTMModel, VisualConcatLSTMModel

seed = 0
rn.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# torch.backends.cudnn.deterministic = True

te = 'te'

# load the data
glove_filename = '../glove.840B.300d.txt'
dev_filename = '../vsnli/VSNLI_1.0_dev.tsv'
train_filename = '../vsnli/VSNLI_1.0_train.tsv'
test_filename = '../vsnli/VSNLI_1.0_test.tsv'
test_hard_filename = '../vsnli/VSNLI_1.0_test_hard.tsv'

# hyperparameters
lr = 0.001
epochs = 50
num_labels = 3
batch_size = 128
print_freq = 1000
hidden_size = 512
max_vocab = 300000
embedding_size = 300

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('metadata' + ".index", mode="rb") as out_file:
    metadata = pickle.load(out_file)

token2id = metadata["token2id"] 
id2token = metadata["id2token"] 
label2id = metadata["label2id"] 
id2label = metadata["id2label"]

num_labels = len(label2id)  
embeddings = torch.tensor(metadata["embeddings"], dtype=torch.float32, requires_grad=True)

with open(f'../data/{te}/' + "dev.pkl", mode="rb") as out_file: dev = pickle.load(out_file)
with open(f'../data/{te}/' + "train.pkl", mode="rb") as out_file: train = pickle.load(out_file)
with open(f'../data/{te}/' + "test.pkl", mode="rb") as out_file: test = pickle.load(out_file)
with open(f'../data/{te}/' + "test_hard.pkl", mode="rb") as out_file: test_hard = pickle.load(out_file)

if te == 'vte':
    with open(f'../data/{te}/vgg16_bn/' + "dev.npy", mode="rb") as out_file: dev_image_feats = np.load(out_file)
    with open(f'../data/{te}/vgg16_bn/' + "train.npy", mode="rb") as out_file: train_image_feats = np.load(out_file)
    with open(f'../data/{te}/vgg16_bn/' + "test.npy", mode="rb") as out_file: test_image_feats = np.load(out_file)
    with open(f'../data/{te}/vgg16_bn/' + "test_hard.npy", mode="rb") as out_file: test_hard_image_feats = np.load(out_file)

print("\n++++++++++ DATA STATISTICS ++++++++++")
print(f"Number of labels: {num_labels}")     
print(f"Dev size: {len(dev['labels'])}")
print(f"Train size: {len(train['labels'])}")
print(f"Test size: {len(test['labels'])}")
print(f"Test_hard size: {len(test_hard['labels'])}")
print("++++++++++ DATA STATISTICS ++++++++++\n")

if te == 'te':
    train_dataset = LSTMDataset(train['padded_premises'], train['padded_hypotheses'], train['labels'])
    dev_dataset = LSTMDataset(dev['padded_premises'], dev['padded_hypotheses'], dev['labels'])
    test_dataset = LSTMDataset(test['padded_premises'], test['padded_hypotheses'], test['labels'])
elif te == 'vte':
    train_dataset = VisualLSTMDataset(train['padded_premises'], train['padded_hypotheses'], train['labels'], train['image_names'], train_image_feats)
    dev_dataset = VisualLSTMDataset(dev['padded_premises'], dev['padded_hypotheses'], dev['labels'], dev['image_names'], dev_image_feats)
    test_dataset = VisualLSTMDataset(test['padded_premises'], test['padded_hypotheses'], test['labels'], test['image_names'], test_image_feats)


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
test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=test_dataset.collater,
            shuffle=False,
        )

model = VanillaLSTMModel(embeddings, embedding_size, hidden_size, num_labels)
# model = ConcatLSTMModel(embeddings, embedding_size, hidden_size, num_labels)
# model = VisualConcatLSTMModel(embeddings, embedding_size, hidden_size, num_labels)

# NOTE: Uncomment the following condition if you want to use DataParallel
# if torch.cuda.device_count() > 0:
#     print("+++++ More than one GPUs available! Applying DataParallel ...")
#     model = nn.DataParallel(model)

model = model.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

stop_criterion = 0
best_dev_accuracy = 0.0

for epoch in range(epochs):
    print(f"\n+++++ Epoch {epoch+1} starting training ...")
    
    start = time.time()
    total_loss = 0

    if stop_criterion >= 3:
        print("+++++ Already reached stopping criterion! Stopping training...")
        break

    for i, batch in tqdm(enumerate(train_dataloader), desc='Training loop', ncols=100, total=len(train_dataloader)):
        labels = batch['labels'].to(device)
        premises = batch['premises'].to(device)
        hypotheses = batch['hypotheses'].to(device)

        if te == 'te':
            out = model(premises, hypotheses)
        elif te == 'vte':
            features = batch['features'].to(device)
            out = model(premises, hypotheses, features)

        loss = criterion(out, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        total_loss += loss.item()

        # NOTE: To create the computation graph, uncomment the following line
        # make_dot(model(premises, hypotheses), params=dict(list(model.named_parameters()))).render("attached", format='png')

        if (i+1) % print_freq == 0:
            print(f"({i+1} batches completed) Loss: {total_loss/(i+1):.4f}")
    
    end = time.time()
    print(f"Time taken for training: {(end - start) // 60} m {(end - start) % 60:.4f} s")

    # save the checkpoint
    # TODO: Add later an option key to store the hyperparameters in ckpt
    save_dir = '.models/concat-lstm/'
    save_path = save_dir + f'epoch_{epoch+1}.ckpt'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("+++++ Saving checkpoint ...")
    torch.save({'weights': model.state_dict(), 'epoch': epoch+1}, save_path)
            
    # Evaluate on dev dataset
    with torch.no_grad():
        tl, correct, total = 0, 0, 0
        for _, batch in enumerate(dev_dataloader):
            labels = batch['labels'].to(device)
            premises = batch['premises'].to(device)
            hypotheses = batch['hypotheses'].to(device)

            if te == 'te':
                out = model(premises, hypotheses)
            elif te == 'vte':
                features = batch['features'].to(device)
                out = model(premises, hypotheses, features)

            loss = criterion(out, labels)
            tl += loss.item()

            out = torch.nn.functional.softmax(out, dim=1)
            _, predicted = torch.max(out, dim=1)
            correct += (predicted == labels).sum().item()
            total += len(labels)

        dev_accuracy = correct/total * 100

        # logic for stopping criterion
        if best_dev_accuracy < dev_accuracy:
            best_dev_accuracy = dev_accuracy
            stop_criterion = 0
        else:
            print("Accuracy on dev set did not increase ...")
            stop_criterion += 1

        print(f"Dev loss: {tl/len(dev_dataloader):.4f} Dev accuracy: {correct/total * 100:.4f}")
    
    # Evaluate on test dataset
    with torch.no_grad():
        tl, correct, total = 0, 0, 0

        for _, batch in enumerate(test_dataloader):
            labels = batch['labels'].to(device)
            premises = batch['premises'].to(device)
            hypotheses = batch['hypotheses'].to(device)

            if te == 'te':
                out = model(premises, hypotheses)
            elif te == 'vte':
                features = batch['features'].to(device)
                out = model(premises, hypotheses, features)

            loss = criterion(out, labels)
            tl += loss.item()

            out = torch.nn.functional.softmax(out, dim=1)
            _, predicted = torch.max(out, dim=1)
            correct += (predicted == labels).sum().item()
            total += len(labels)
        test_accuracy = correct/total * 100
        print(f"Test loss: {tl/len(test_dataloader):.4f} Test accuracy: {correct/total * 100:.4f}")