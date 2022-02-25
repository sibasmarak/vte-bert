import torch
import pickle
import time, os
import numpy as np
import random as rn
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

from data import BERTDataset, VisualBERTDataset
from models import ConcatBERTModel, VanillaBERTModel, VisualConcatBERTModel

# NOTE: Uncomment the following for creating the computation graph
# from torchviz import make_dot, make_dot_from_trace
# import matplotlib.pyplot as plt
# from graphviz import Source

seed = 0
rn.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
# torch.backends.cudnn.deterministic = True

# hyperparameters
epochs = 30
num_labels = 3
print_freq = 500
hidden_size = 512

lr = 1e-5
wd = 5e-3
batch_size = 32
te = 'vte' # te, vte etc.
model_path = 'grounding-models32_1' # models, grounding-models etc.
bert = 'bert' # bert, roberta etc.
bert_type = 'tiny' # tinier, tiny, small, base, pretrained etc.
vision_model = 'resnet50' # vgg16_bn, resnet50
modeling = 'visualconcat' # vanilla, concat, visualconcat

print(te, model_path, bert, bert_type, vision_model, modeling)
hyperparameter = {'lr': lr, 'epochs': epochs, 'hidden_size': hidden_size, 'batch_size': batch_size, 'weight_decay': wd}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('metadata' + ".index", mode="rb") as out_file:
	metadata = pickle.load(out_file)

token2id = metadata["token2id"] 
id2token = metadata["id2token"] 
label2id = metadata["label2id"] 
id2label = metadata["id2label"]     
num_labels = len(label2id)  

with open(f'../data/{te}/' + "dev.pkl", mode="rb") as out_file: dev = pickle.load(out_file)
with open(f'../data/{te}/' + "train.pkl", mode="rb") as out_file: train = pickle.load(out_file)
with open(f'../data/{te}/' + "test.pkl", mode="rb") as out_file: test = pickle.load(out_file)
with open(f'../data/{te}/' + "test_hard.pkl", mode="rb") as out_file: test_hard = pickle.load(out_file)

if te == 'vte':
	with open(f'../data/{te}/{vision_model}/' + "dev.npy", mode="rb") as out_file: dev_image_feats = np.load(out_file)
	with open(f'../data/{te}/{vision_model}/' + "train.npy", mode="rb") as out_file: train_image_feats = np.load(out_file)
	with open(f'../data/{te}/{vision_model}/' + "test.npy", mode="rb") as out_file: test_image_feats = np.load(out_file)
	with open(f'../data/{te}/{vision_model}/' + "test_hard.npy", mode="rb") as out_file: test_hard_image_feats = np.load(out_file)


print("\n++++++++++ DATA STATISTICS ++++++++++")
print(f"Number of labels: {num_labels}")     
print(f"Dev size: {len(dev['labels'])}")
print(f"Train size: {len(train['labels'])}")
print(f"Test size: {len(test['labels'])}")
print(f"Test_hard size: {len(test_hard['labels'])}")
print("++++++++++ DATA STATISTICS ++++++++++\n")

if te == 'te':
	train_dataset = BERTDataset(train['original_premises'], train['original_hypotheses'], train['labels'])
	dev_dataset = BERTDataset(dev['original_premises'], dev['original_hypotheses'], dev['labels'])
	test_dataset = BERTDataset(test['original_premises'], test['original_hypotheses'], test['labels'])
elif te == 'vte':
	train_dataset = VisualBERTDataset(train['original_premises'], train['original_hypotheses'], train['labels'], train['image_names'], train_image_feats)
	dev_dataset = VisualBERTDataset(dev['original_premises'], dev['original_hypotheses'], dev['labels'], dev['image_names'], dev_image_feats)
	test_dataset = VisualBERTDataset(test['original_premises'], test['original_hypotheses'], test['labels'], test['image_names'], test_image_feats)
	image_embedding_size = test_image_feats.squeeze().shape[1]
	print(f'+++++ Image embedding size from {vision_model}: {image_embedding_size}')

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

model = model.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=wd)
criterion = nn.CrossEntropyLoss()

stop_criterion = 0
best_dev_accuracy = 0.0

print(f"+++++ Training with BERT {bert_type} model ...")
for epoch in range(epochs):
	print(f"\n+++++ Epoch {epoch+1} starting training ...")
	
	start = time.time()
	total_loss = 0

	if stop_criterion >= 3:
		print("+++++ Already reached stopping criterion! Stopping training...")
		break

	for i, batch in enumerate(train_dataloader):
		
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
	save_dir = f'.{model_path}/{modeling}/{bert}-{bert_type}/'
	save_path = save_dir + f'epoch_{epoch+1}.ckpt'

	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	print("+++++ Saving checkpoint ...")
	torch.save({'weights': model.state_dict(), 'epoch': epoch+1, 'hyperparameter': hyperparameter}, save_path)
			
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