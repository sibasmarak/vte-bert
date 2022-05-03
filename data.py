import torch, clip
from PIL import Image
import pickle
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer

class ImageReader:
	def __init__(self, img_names, img_features):
		self.img_names = img_names
		self.img_features = img_features

		self.img_names_features = {filename: features for filename, features in zip(img_names, img_features)}

	def get_features(self, image_name):
		return self.img_names_features[image_name]

class BERTDataset(Dataset):
	def __init__(self, premises, hypotheses, labels, 
					bert_type='bert-base-uncased', max_length=64):
		self.premises = list(premises)
		self.hypotheses = list(hypotheses)
		self.labels = list(labels)
		
		self.bert_type = bert_type

		if self.bert_type == 'bert-base-uncased':
			self.model_dir = '/home/bt2/18CS10069/btp2/bert-base-uncased/'

		self.tokenizer = BertTokenizer.from_pretrained(self.model_dir)

	def __len__(self):
		return len(self.labels)

	def encode(self, text, max_length):
		return self.tokenizer.encode(text, padding="max_length", 
			max_length=max_length, truncation=True, return_tensors = "pt")

	def __getitem__(self, x):
		return (self.premises[x], 
					self.hypotheses[x], 
					torch.tensor(self.labels[x]).long())

	def collater(self, items):

		for item_idx in range(len(items)):
			items[item_idx] = list(items[item_idx])
			items[item_idx][0] = self.encode(items[item_idx][0], max_length=64)
			items[item_idx][1] = self.encode(items[item_idx][1], max_length=64)

		batch = {
			'premises': torch.stack([x[0][0] for x in items], dim=0),
			'hypotheses': torch.stack([x[1][0] for x in items], dim=0),
			'labels': torch.stack([x[2] for x in items], dim=0),
		}
		return batch

class LSTMDataset(Dataset):
	def __init__(self, premises, hypotheses, labels):
		self.premises = premises
		self.hypotheses = hypotheses
		self.labels = labels
		
	def __len__(self):
		return len(self.labels)

	def __getitem__(self, x):
		return (torch.tensor(self.premises[x]), 
					torch.tensor(self.hypotheses[x]), 
					torch.tensor(self.labels[x]))

	def collater(self, items):
		premise_lengths = np.array([len(item[0]) for item in items])
		hypothesis_lengths = np.array([len(item[1]) for item in items])
		
		premise_max_length = np.max(premise_lengths)
		hypothesis_max_length = np.max(hypothesis_lengths)

		# pad each item till maximum length of the batch
		for item_idx in range(len(items)):
			items[item_idx] = list(items[item_idx])
			items[item_idx][0] = torch.tensor(items[item_idx][0].tolist() + [0] * (premise_max_length - len(items[item_idx][0])))
			items[item_idx][1] = torch.tensor(items[item_idx][1].tolist() + [0] * (hypothesis_max_length - len(items[item_idx][1])))

		batch = {
			'premises': torch.stack([x[0] for x in items], dim=0),
			'hypotheses': torch.stack([x[1] for x in items], dim=0),
			'labels': torch.stack([x[2] for x in items], dim=0),
		}

		return batch

class VisualLSTMDataset(Dataset):
	def __init__(self, premises, hypotheses, labels, image_names, image_features):
		self.premises = premises
		self.hypotheses = hypotheses
		self.labels = labels
		self.image_names = image_names
		self.image_features = image_features

		self.im_reader = ImageReader(image_names, image_features)
		
	def __len__(self):
		return len(self.labels)

	def __getitem__(self, x):
		image_features = self.im_reader.get_features(self.image_names[x])
		return (torch.tensor(self.premises[x]), 
					torch.tensor(self.hypotheses[x]), 
					torch.tensor(self.labels[x]),
					torch.tensor(image_features))

	def collater(self, items):
		premise_lengths = np.array([len(item[0]) for item in items])
		hypothesis_lengths = np.array([len(item[1]) for item in items])
		
		premise_max_length = np.max(premise_lengths)
		hypothesis_max_length = np.max(hypothesis_lengths)

		# pad each item till maximum length of the batch
		for item_idx in range(len(items)):
			items[item_idx] = list(items[item_idx])
			items[item_idx][0] = torch.tensor(items[item_idx][0].tolist() + [0] * (premise_max_length - len(items[item_idx][0])))
			items[item_idx][1] = torch.tensor(items[item_idx][1].tolist() + [0] * (hypothesis_max_length - len(items[item_idx][1])))

		batch = {
			'premises': torch.stack([x[0] for x in items], dim=0),
			'hypotheses': torch.stack([x[1] for x in items], dim=0),
			'labels': torch.stack([x[2] for x in items], dim=0),
			'features': torch.stack([x[3][0] for x in items], dim=0),
		}

		return batch

class VisualBERTDataset(Dataset):
	def __init__(self, premises, hypotheses, labels, image_names, image_features, 
					bert_type='bert-base-uncased', max_length=64):

		self.premises = list(premises)
		self.hypotheses = list(hypotheses)
		self.labels = list(labels)
		self.image_names = list(image_names)
		self.image_features = image_features

		self.im_reader = ImageReader(image_names, image_features)
		
		self.bert_type = bert_type

		if self.bert_type == 'bert-base-uncased':
			self.model_dir = '/home/bt2/18CS10069/btp2/bert-base-uncased/'

		self.tokenizer = BertTokenizer.from_pretrained(self.model_dir)
		
	def __len__(self):
		return len(self.labels)

	def encode(self, text, max_length):
		return self.tokenizer.encode(text, padding="max_length", 
			max_length=max_length, truncation=True, return_tensors = "pt")

	def __getitem__(self, x):
		image_features = self.im_reader.get_features(self.image_names[x])
		return (self.premises[x], 
				self.hypotheses[x], 
				torch.tensor(self.labels[x]).long(),
				torch.tensor(image_features))

	def collater(self, items):
		for item_idx in range(len(items)):
			items[item_idx] = list(items[item_idx])
			items[item_idx][0] = self.encode(items[item_idx][0], max_length=64)
			items[item_idx][1] = self.encode(items[item_idx][1], max_length=64)

		batch = {
			'premises': torch.stack([x[0][0] for x in items], dim=0),
			'hypotheses': torch.stack([x[1][0] for x in items], dim=0),
			'labels': torch.stack([x[2] for x in items], dim=0),
			'features': torch.stack([x[3][0] for x in items], dim=0),
		}

		return batch



class MultiModalDataset(Dataset):
	def __init__(self, premises, hypotheses, labels, image_names, vision_modelname, 
					bert_type='bert-base-uncased', max_length=64, use_timm=False, data='snli'):

		self.premises = premises
		self.hypotheses = hypotheses
		self.labels = labels
		self.image_names = image_names
		self.data = data
		self.vision_modelname = vision_modelname
		self.use_timm = use_timm

		if not self.use_timm:
			_, self.preprocess = clip.load(self.vision_modelname)
		
		
		self.bert_type = bert_type

		if self.bert_type == 'bert-base-uncased':
			self.model_dir = '/home/bt2/18CS10069/btp2/bert-base-uncased/'

		self.tokenizer = BertTokenizer.from_pretrained(self.model_dir)
		
	def __len__(self):
		return len(self.labels)

	def encode(self, text, max_length):
		return self.tokenizer.encode(text, padding="max_length", 
			max_length=max_length, truncation=True, return_tensors = "pt")

	def __getitem__(self, x):


		if self.data == 'snli':
			orig_im = Image.open(f'../flickr30k-images/{self.image_names[x]}')
		elif self.data == 'tweet':
			orig_im = Image.open(f'../{self.image_names[x]}')

		if self.use_timm:
			im = orig_im.resize((224, 224))
			im = np.array(im).reshape((3, 224, 224))
			im = im.astype('float32')
			im /= 255

			im = torch.tensor(im).unsqueeze(0)

		else:
			im = self.preprocess(orig_im).unsqueeze(0)

		return (self.premises[x], 
				self.hypotheses[x], 
				torch.tensor(self.labels[x]).long(),
				im)

	def collater(self, items):
		for item_idx in range(len(items)):
			items[item_idx] = list(items[item_idx])
			items[item_idx][0] = self.encode(items[item_idx][0], max_length=64)
			items[item_idx][1] = self.encode(items[item_idx][1], max_length=64)

		batch = {
			'encoded_premises': torch.stack([x[0][0] for x in items], dim=0),
			'encoded_hypotheses': torch.stack([x[1][0] for x in items], dim=0),
			'labels': torch.stack([x[2] for x in items], dim=0),
			'ims': torch.cat([x[3] for x in items], dim=0),
		}
		return batch