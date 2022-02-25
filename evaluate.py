import torch
import pickle
import time, os
import numpy as np
import random as rn
import torch.nn as nn
from collections import OrderedDict, defaultdict
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from prettytable import PrettyTable
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score

from data import BERTDataset, VisualBERTDataset
from models import ConcatBERTModel, VanillaBERTModel, VisualConcatBERTModel

seed = 0
rn.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# torch.backends.cudnn.deterministic = True

# hyperparameters
lr = 1e-5
epochs = 30
num_labels = 3
batch_size = 32
print_freq = 500
hidden_size = 512
bert = 'bert' # bert, roberta etc.
vision_model = 'resnet50' # vgg16_bn, resnet50

# NOTE: necessary for eval
te = 'te' # te, vte etc.
bert_type = 'tiny' # tiny, small, base, pretrained etc.
modeling = 'concat' # vanilla, concat, visualconcat
modelpath = '/home/bt2/18CS10069/btp2/src/.models256_8_1/concat/bert-tiny/epoch_3.ckpt'

print(te, bert, bert_type, vision_model, modeling)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the data
with open('metadata' + ".index", mode="rb") as out_file:
	metadata = pickle.load(out_file)

token2id = metadata["token2id"] 
id2token = metadata["id2token"] 
label2id = metadata["label2id"] 
id2label = metadata["id2label"]     
num_labels = len(label2id)  

with open(f'../data/{te}/' + "dev.pkl", mode="rb") as out_file: dev = pickle.load(out_file)
with open(f'../data/{te}/' + "test.pkl", mode="rb") as out_file: test = pickle.load(out_file)
with open(f'../data/{te}/' + "test_hard.pkl", mode="rb") as out_file: test_hard = pickle.load(out_file)
with open(f'../data/{te}/' + "test_lexical.pkl", mode="rb") as out_file: test_lexical = pickle.load(out_file)

if te == 'vte':
	with open(f'../data/{te}/{vision_model}/' + "dev.npy", mode="rb") as out_file: dev_image_feats = np.load(out_file)
	with open(f'../data/{te}/{vision_model}/' + "test.npy", mode="rb") as out_file: test_image_feats = np.load(out_file)
	with open(f'../data/{te}/{vision_model}/' + "test_hard.npy", mode="rb") as out_file: test_hard_image_feats = np.load(out_file)
	with open(f'../data/{te}/{vision_model}/' + "test_lexical.npy", mode="rb") as out_file: test_lexical_image_feats = np.load(out_file)


print("\n++++++++++ DATA STATISTICS ++++++++++")
print(f"Number of labels: {num_labels}")     
print(f"Dev size: {len(dev['labels'])}")
print(f"Test size: {len(test['labels'])}")
print(f"Test_hard size: {len(test_hard['labels'])}")
print(f"Test_lexical size: {len(test_lexical['labels'])}")
print("++++++++++ DATA STATISTICS ++++++++++\n")

if te == 'te':
	dev_dataset = BERTDataset(dev['original_premises'], dev['original_hypotheses'], dev['labels'])
	test_dataset = BERTDataset(test['original_premises'], test['original_hypotheses'], test['labels'])
	test_hard_dataset = BERTDataset(test_hard['original_premises'], test_hard['original_hypotheses'], test_hard['labels'])
	test_lexical_dataset = BERTDataset(test_lexical['original_premises'], test_lexical['original_hypotheses'], test_lexical['labels'])
elif te == 'vte':
	dev_dataset = VisualBERTDataset(dev['original_premises'], dev['original_hypotheses'], dev['labels'], dev['image_names'], dev_image_feats)
	test_dataset = VisualBERTDataset(test['original_premises'], test['original_hypotheses'], test['labels'], test['image_names'], test_image_feats)
	test_hard_dataset = VisualBERTDataset(test_hard['original_premises'], test_hard['original_hypotheses'], test_hard['labels'], test_hard['image_names'], test_hard_image_feats)
	test_lexical_dataset = VisualBERTDataset(test_lexical['original_premises'], test_lexical['original_hypotheses'], test_lexical['labels'], test_lexical['image_names'], test_lexical_image_feats)
	image_embedding_size = test_image_feats.squeeze().shape[1]
	print(f'+++++ Image embedding size from {vision_model}: {image_embedding_size}')

test_hard_dataloader = DataLoader(
			test_hard_dataset,
			batch_size=batch_size,
			collate_fn=test_hard_dataset.collater,
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
test_lexical_dataloader = DataLoader(
			test_lexical_dataset,
			batch_size=batch_size,
			collate_fn=test_lexical_dataset.collater,
			shuffle=False,
		)

if modeling == 'vanilla':
	model = VanillaBERTModel(num_labels, hidden_size, bert_type=bert_type)
elif modeling == 'concat':
	model = ConcatBERTModel(num_labels, hidden_size, bert_type=bert_type)
elif modeling == 'visualconcat':
	model = VisualConcatBERTModel(num_labels, hidden_size, bert_type=bert_type, image_embedding_size=image_embedding_size)

# NOTE: Uncomment the following condition if you want to use DataParallel
if torch.cuda.device_count() > 0:
	print("+++++ More than one GPUs available! Applying DataParallel ...")
	model = nn.DataParallel(model)

# NOTE: code to load from checkpoint
if modelpath != None:
    A = torch.load(modelpath, map_location=device)
    load_state_dict = A['weights']
    load_prefix = list(load_state_dict.keys())[0][:6]
    new_state_dict = {}
    for key in load_state_dict:
        value = load_state_dict[key]
        # Multi-GPU state dict has the prefix 'module.' appended in front of each key
        if torch.cuda.device_count() > 1:
            if load_prefix != 'module':
                new_key = 'module.' + key
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        else:
            if load_prefix == 'module':
                new_key = key[7:]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
    model.load_state_dict(new_state_dict)


model = model.to(device)
criterion = nn.CrossEntropyLoss()
model.eval()

# Evaluate on test dataset
index_loader_mapper = {0: 'dev', 1:'test', 2: 'test_hard', 3: 'test_lexical'}
with torch.no_grad():

    for i, dataloader in enumerate([dev_dataloader, test_dataloader, test_hard_dataloader, test_lexical_dataloader]):
        print(f"\n+++++ Evaluate on {index_loader_mapper[i]} set ...")

        tl, correct, total, ACC, F1 = 0, 0, 0, 0, 0
        labelwise_correct = defaultdict(lambda : 0)
        labelwise_total = defaultdict(lambda : 0)

        y_true, y_pred = [], []
        
        for _, batch in enumerate(dataloader):
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

            y_true.extend(labels.detach().cpu().numpy())
            y_pred.extend(predicted.detach().cpu().numpy())

            labelwise_correct[label2id['entailment']] += torch.logical_and(torch.eq(predicted, label2id['entailment']), 
                                                                    torch.eq(labels, label2id['entailment'])).sum().item()
            labelwise_correct[label2id['neutral']] += torch.logical_and(torch.eq(predicted, label2id['neutral']),
                                                                    torch.eq(labels, label2id['neutral'])).sum().item()
            labelwise_correct[label2id['contradiction']] += torch.logical_and(torch.eq(predicted, label2id['contradiction']),
                                                                    torch.eq(labels, label2id['contradiction'])).sum().item()

            labelwise_total[label2id['entailment']] += (labels == label2id['entailment']).sum().item()
            labelwise_total[label2id['neutral']] += (labels == label2id['neutral']).sum().item()
            labelwise_total[label2id['contradiction']] += (labels == label2id['contradiction']).sum().item()

        accuracy = correct/total * 100
        ACC = balanced_accuracy_score(y_true, y_pred)

        print(f"Test loss: {tl/len(dataloader):.4f}")
        table = PrettyTable()
        table.field_names = [f"label name", "accuracy"]
        for label, value in label2id.items():
            acc = np.round(labelwise_correct[value]/labelwise_total[value] * 100, decimals=4)
            table.add_row([label, acc])
        table.add_row(['total', np.round(accuracy, decimals=4)])
        table.add_row(['balanced_total', np.round(ACC * 100, decimals=4)])
        print(table)

        # uncomment the following to obtain the data distribution
        # table = PrettyTable()
        # table.field_names = [f"label name", "total"]
        # for label, value in label2id.items():
        #     number = labelwise_total[value]
        #     table.add_row([label, number])
        # print(table)