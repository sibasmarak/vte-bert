import clip, torch, time, timm
import torch.nn as nn
from collections import OrderedDict

from transformers import BertTokenizer, BertModel, BertConfig

class VanillaBERTModel(nn.Module):
	def __init__(self, num_labels, hidden_size, bert_type='tiny', 
							dropout_ratio=0.5, output_size=768):
		super(VanillaBERTModel, self).__init__()

		self.bert_type = bert_type
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.dropout_ratio = dropout_ratio

		if self.bert_type == 'tinier':
			self.config = BertConfig(num_hidden_layers=2, max_position_embeddings=64, intermediate_size=128, num_attention_heads=2, output_attentions=True) # Bert tinier config
		elif self.bert_type == 'tiny_middle':
			self.config = BertConfig(num_hidden_layers=2, max_position_embeddings=128, intermediate_size=256, num_attention_heads=4, output_attentions=True) # Bert tinier config
		elif self.bert_type == 'tiny':
			self.config = BertConfig(num_hidden_layers=4, max_position_embeddings=128, intermediate_size=256, num_attention_heads=4, output_attentions=True) # Bert tiny config
		elif self.bert_type == 'small':
			self.config = BertConfig(num_hidden_layers=6, max_position_embeddings=512, intermediate_size=1024, num_attention_heads=4, output_attentions=True) # Bert small config
		elif self.bert_type == 'base':
			self.config = BertConfig(num_hidden_layers=6, max_position_embeddings=512, intermediate_size=2048, num_attention_heads=8, output_attentions=True) # Bert base config
		
		self.model = BertModel(config=self.config)
		if self.bert_type == 'pretrained':
			self.model = BertModel.from_pretrained('~/btp2/bert-base-uncased')


		self.fc1 = nn.Linear(output_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.fc3 = nn.Linear(hidden_size, num_labels)

		self.activation = nn.ReLU()
		self.dropout = nn.Dropout(p=self.dropout_ratio)

		self.fc = nn.Sequential(
			self.fc1,
			self.activation,
			self.dropout,
			self.fc2,
			self.activation,
			self.dropout,
			self.fc3
		)

	def forward(self, encoded_premises, encoded_hypotheses):

		encoding = self.model(input_ids=encoded_hypotheses)
		out = encoding.pooler_output
		out = self.fc(out)
		return out

class ConcatBERTModel(nn.Module):
	def __init__(self, num_labels, hidden_size, bert_type='tiny', dropout_ratio=0.5, output_size=768):
		super(ConcatBERTModel, self).__init__()

		self.bert_type = bert_type
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.dropout_ratio = dropout_ratio
		self.config = None

		if self.bert_type == 'tinier':
			self.config = BertConfig(num_hidden_layers=2, max_position_embeddings=64, intermediate_size=128, num_attention_heads=2, output_attentions=True) # Bert tinier config
		elif self.bert_type == 'tiny_middle':
			self.config = BertConfig(num_hidden_layers=2, max_position_embeddings=128, intermediate_size=256, num_attention_heads=4, output_attentions=True) # Bert tinier config
		elif self.bert_type == 'tiny':
			self.config = BertConfig(num_hidden_layers=4, max_position_embeddings=128, intermediate_size=256, num_attention_heads=4, output_attentions=True) # Bert tiny config
		elif self.bert_type == 'small':
			self.config = BertConfig(num_hidden_layers=6, max_position_embeddings=512, intermediate_size=1024, num_attention_heads=4, output_attentions=True) # Bert small config
		elif self.bert_type == 'base':
			self.config = BertConfig(num_hidden_layers=6, max_position_embeddings=512, intermediate_size=2048, num_attention_heads=8, output_attentions=True) # Bert base config
		
		if self.config:
			self.bert_premises = BertModel(config=self.config)
			self.bert_hypotheses = BertModel(config=self.config)
		elif self.bert_type == 'pretrained':
			self.bert_premises = BertModel.from_pretrained('/home/bt2/18CS10069/btp2/bert-base-uncased')
			self.bert_hypotheses = BertModel.from_pretrained('/home/bt2/18CS10069/btp2/bert-base-uncased')

		self.fc1 = nn.Linear(2 * output_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.fc3 = nn.Linear(hidden_size, num_labels)

		self.activation = nn.ReLU()
		self.dropout = nn.Dropout(p=self.dropout_ratio)

		self.fc = nn.Sequential(
			self.fc1,
			self.activation,
			self.dropout,
			self.fc2,
			self.activation,
			self.dropout,
			self.fc3
		)

	def forward(self, encoded_premises, encoded_hypotheses):
		encoding_premises = self.bert_premises(input_ids=encoded_premises)
		encoding_hypotheses = self.bert_hypotheses(input_ids=encoded_hypotheses)
		out = torch.cat((encoding_premises.pooler_output, encoding_hypotheses.pooler_output), dim=1)
		out = self.fc(out)
		return out

class VanillaLSTMModel(nn.Module):
	def __init__(self, embeddings, embedding_size, hidden_size, num_labels, dropout_ratio = 0.5):
		super(VanillaLSTMModel, self).__init__()

		self.embedding_size = embedding_size
		self.hidden_size = hidden_size
		self.num_labels = num_labels
		self.dropout_ratio = dropout_ratio

		self.emb = nn.Embedding.from_pretrained(embeddings, freeze=False)
		self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, batch_first=True)

		self.fc1 = nn.Linear(hidden_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.fc3 = nn.Linear(hidden_size, num_labels)
		self.activation = nn.ReLU()
		self.dropout = nn.Dropout(p=self.dropout_ratio)

		self.fc = nn.Sequential(
			self.fc1,
			self.activation,
			self.dropout,
			self.fc2,
			self.activation,
			self.dropout,
			self.fc3
		)

	def forward(self, premises, hypotheses):
		out = self.emb(hypotheses)
		_, (h, _) = self.lstm(self.dropout(out)) # bs x seq len x hidden dim, num_layer * directions x bs x hidden dim
		out = h[-1]
		out = self.fc(out)
		return out

class ConcatLSTMModel(nn.Module):
	def __init__(self, embeddings, embedding_size, hidden_size, num_labels, dropout_ratio = 0.5, num_layers=1):
		super(ConcatLSTMModel, self).__init__()

		self.embedding_size = embedding_size
		self.hidden_size = hidden_size
		self.num_labels = num_labels
		self.dropout_ratio = dropout_ratio
		self.num_layers = num_layers

		self.emb = nn.Embedding.from_pretrained(embeddings, freeze=False)

		self.lstm_hyp = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=self.num_layers, batch_first=True, dropout=self.dropout_ratio)
		self.lstm_prem = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=self.num_layers, batch_first=True, dropout=self.dropout_ratio)

		self.fc1 = nn.Linear(2 * hidden_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.fc3 = nn.Linear(hidden_size, num_labels)
		self.activation = nn.ReLU()
		self.dropout = nn.Dropout(p=self.dropout_ratio)

		self.fc = nn.Sequential(
			self.fc1,
			self.activation,
			self.dropout,
			self.fc2,
			self.activation,
			self.dropout,
			self.fc3
		)

	def forward(self, premises, hypotheses):

		hyp_embedding_layer = self.emb(hypotheses)
		prem_embedding_layer = self.emb(premises)

		_, (hyp_hn, _) = self.lstm_hyp(self.dropout(hyp_embedding_layer))
		_, (prem_hn, _) = self.lstm_prem(self.dropout(prem_embedding_layer))

		hyp = self.dropout(hyp_hn[-1])
		prem = self.dropout(prem_hn[-1])

		out = torch.cat((prem, hyp), dim=1)
		out = self.fc(out)
		return out

class VisualConcatLSTMModel(nn.Module):
	def __init__(self, embeddings, embedding_size, hidden_size, num_labels, num_layers=1, image_embedding_size=4096, dropout_ratio = 0.5):

		super(VisualConcatLSTMModel, self).__init__()

		self.embedding_size = embedding_size
		self.hidden_size = hidden_size
		self.num_labels = num_labels
		self.dropout_ratio = dropout_ratio
		self.num_layers = num_layers

		self.im_projection = nn.Linear(image_embedding_size, hidden_size)
		self.ph_projection = nn.Linear(hidden_size, hidden_size)

		self.emb = nn.Embedding.from_pretrained(embeddings, freeze=False)

		self.lstm_hyp = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=self.num_layers, batch_first=True, dropout=self.dropout_ratio)
		self.lstm_prem = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=self.num_layers, batch_first=True, dropout=self.dropout_ratio)

		self.fc1 = nn.Linear(2 * hidden_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.fc3 = nn.Linear(hidden_size, num_labels)
		self.activation = nn.ReLU()
		self.dropout = nn.Dropout(p=self.dropout_ratio)

		self.fc = nn.Sequential(
			self.fc1,
			self.activation,
			self.dropout,
			self.fc2,
			self.activation,
			self.dropout,
			self.fc3
		)

	def forward(self, premises, hypotheses, features):
		proj_features = self.im_projection(features)

		hyp_embedding_layer = self.emb(hypotheses)
		prem_embedding_layer = self.emb(premises)

		_, (hyp_hn, _) = self.lstm_hyp(self.dropout(hyp_embedding_layer))
		_, (prem_hn, _) = self.lstm_prem(self.dropout(prem_embedding_layer))

		hyp = self.dropout(hyp_hn[-1])
		prem = self.dropout(prem_hn[-1])

		prem_hn = self.ph_projection(prem)
		hyp_hn = self.ph_projection(hyp)

		prem_hn = prem_hn * proj_features
		hyp_hn = hyp_hn * proj_features

		out = torch.cat((prem, hyp), dim=1)
		out = self.fc(out)
		return out

class VisualConcatBERTModel(nn.Module):
	def __init__(self, num_labels, hidden_size, bert_type='tiny', image_embedding_size=4096, dropout_ratio=0.5, output_size=768):
		super(VisualConcatBERTModel, self).__init__()

		self.bert_type = bert_type
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.dropout_ratio = dropout_ratio
		self.config = None

		self.projection = nn.Linear(image_embedding_size, hidden_size)
		self.ph_projection = nn.Linear(output_size, hidden_size)

		if self.bert_type == 'tinier':
			self.config = BertConfig(num_hidden_layers=2, max_position_embeddings=64, intermediate_size=128, num_attention_heads=2, output_attentions=True) # Bert tinier config
		elif self.bert_type == 'tiny_middle':
			self.config = BertConfig(num_hidden_layers=2, max_position_embeddings=128, intermediate_size=256, num_attention_heads=4, output_attentions=True) # Bert tinier config
		elif self.bert_type == 'tiny':
			self.config = BertConfig(num_hidden_layers=4, max_position_embeddings=128, intermediate_size=256, num_attention_heads=4, output_attentions=True) # Bert tiny config
		elif self.bert_type == 'small':
			self.config = BertConfig(num_hidden_layers=6, max_position_embeddings=512, intermediate_size=1024, num_attention_heads=4, output_attentions=True) # Bert small config
		elif self.bert_type == 'base':
			self.config = BertConfig(num_hidden_layers=6, max_position_embeddings=512, intermediate_size=2048, num_attention_heads=8, output_attentions=True) # Bert base config
		
		if self.config:
			self.bert_premises = BertModel(config=self.config)
			self.bert_hypotheses = BertModel(config=self.config)
		elif self.bert_type == 'pretrained':
			self.bert_premises = BertModel.from_pretrained('/home/bt2/18CS10069/btp2/bert-base-uncased')
			self.bert_hypotheses = BertModel.from_pretrained('/home/bt2/18CS10069/btp2/bert-base-uncased')

		self.fc1 = nn.Linear(2 * hidden_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.fc3 = nn.Linear(hidden_size, num_labels)

		self.activation = nn.ReLU()
		self.dropout = nn.Dropout(p=self.dropout_ratio)

		self.fc = nn.Sequential(
			self.fc1,
			self.activation,
			self.dropout,
			self.fc2,
			self.activation,
			self.dropout,
			self.fc3
		)

	def forward(self, encoded_premises, encoded_hypotheses, features):
		proj_features = self.projection(features)

		encoding_premises = self.bert_premises(input_ids=encoded_premises)
		encoding_hypotheses = self.bert_hypotheses(input_ids=encoded_hypotheses)

		encoding_prem = self.ph_projection(encoding_premises.pooler_output)
		encoding_hyp = self.ph_projection(encoding_hypotheses.pooler_output)

		encoding_prem = encoding_prem * proj_features
		encoding_hyp = encoding_hyp * proj_features

		out = torch.cat((encoding_prem, encoding_hyp), dim=1)
		out = self.fc(out)
		return out

class MultiModalModel(nn.Module):
	def __init__(self, vision_modelname, device, modeling, lm_options, use_timm=False):
		super(MultiModalModel, self).__init__()
		# timm or clip
		self.use_timm = use_timm

		# obtain CLIP-vision model
		self.vision_modelname = vision_modelname
		self.device = device

		if self.use_timm:
			self.vision_model = timm.create_model(self.vision_modelname, pretrained=True)
			self.vision_model.reset_classifier(0)
			self.vision_model = self.vision_model.to(self.device)
		else:
			self.vision_model, _ = clip.load(self.vision_modelname, device=self.device)
			self.vision_model.float()

		# obtain language model
		self.modeling = modeling
		self.lm_options = lm_options 
		if self.modeling == 'vanilla':
			self.language_model = VanillaBERTModel(self.lm_options['num_labels'], 
													self.lm_options['hidden_size'], 
													bert_type=self.lm_options['bert_type'])
		elif self.modeling == 'concat':
			self.language_model = ConcatBERTModel(self.lm_options['num_labels'], 
													self.lm_options['hidden_size'], 
													bert_type=self.lm_options['bert_type'])
		elif self.modeling == 'visualconcat':
			self.language_model = VisualConcatBERTModel(self.lm_options['num_labels'], 
															self.lm_options['hidden_size'], 
															bert_type=self.lm_options['bert_type'], 
															image_embedding_size=self.lm_options['image_embedding_size'])

	def forward(self, batch):
		if self.modeling == 'vanilla' or self.modeling == 'concat':
			out = self.language_model(batch['encoded_premises'].to(self.device), batch['encoded_hypotheses'].to(self.device))

		elif self.modeling == 'visualconcat':
			# batch contains images -- obtain image features
			with torch.no_grad():
				if self.use_timm:
					features = self.vision_model(batch['ims'].to(self.device))
				else:
					features = self.vision_model.encode_image(batch['ims'].to(self.device))

			features = features/torch.linalg.norm(features, ord=2, dim=1, keepdim=True)

			# call forward for language model
			out = self.language_model(batch['encoded_premises'].to(self.device), batch['encoded_hypotheses'].to(self.device), features)

		return out


