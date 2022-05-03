import torch, jsonlines
import pickle
import pandas as pd, os 
import torch.nn as nn
import numpy as np
from collections import OrderedDict

from dataset import load_te_dataset, load_vte_dataset
from embeddings import load_glove

from sklearn.model_selection import train_test_split

torch.manual_seed(0)

# Google Multi-Modal
label_map = {
        "Contradictory": 2, # contradiction
        "Implies": 1, # Entailment
        "NoEntailment": 0 # NoEntailment
    }

df = pd.read_csv("https://github.com/sayakpaul/Multimodal-Entailment-Baseline/raw/main/csvs/tweets.csv")
images_one_paths = []
images_two_paths = []

for idx in range(len(df)):
    current_row = df.iloc[idx]
    id_1 = current_row["id_1"]
    id_2 = current_row["id_2"]
    extentsion_one = current_row["image_1"].split(".") [-1]
    extentsion_two = current_row["image_2"].split(".") [-1]
    
    image_one_path = os.path.join("tweet_images", str(id_1) + f".{extentsion_one}")
    image_two_path = os.path.join("tweet_images", str(id_2) + f".{extentsion_two}")

    images_one_paths.append(image_one_path)
    images_two_paths.append(image_two_path)

df["image_1_path"] = images_one_paths
df["image_2_path"] = images_two_paths
df["label_idx"] = df["label"].apply(lambda x: label_map[x])

train_df, test_df = train_test_split(df, test_size=0.1, stratify=df["label"].values, random_state=0)
train_df, val_df = train_test_split(train_df, test_size=0.05, stratify=train_df["label"].values, random_state=0)

print(f"Total training examples: {len(train_df)}")
print(f"Total validation examples: {len(val_df)}")
print(f"Total test examples: {len(test_df)}")

# NOTE: Dumping TE dataset
# dev set
with open('../data/te/tweet/' + "dev.pkl", mode="wb") as out_file:
    pickle.dump(
        {
            "labels": val_df['label_idx'],
            "original_premises": val_df['text_1'],
            "original_hypotheses": val_df['text_2']
        },
        out_file
    )

with open('../data/vte/tweet/' + "dev.pkl", mode="wb") as out_file:
    pickle.dump(
        {
            "image_names": val_df['image_1_path'],
            "labels": val_df['label_idx'],
            "original_premises": val_df['text_1'],
            "original_hypotheses": val_df['text_2']
        },
        out_file
    )

# train set
with open('../data/te/tweet/' + "train.pkl", mode="wb") as out_file:
    pickle.dump(
        {
            "labels": train_df['label_idx'],
            "original_premises": train_df['text_1'],
            "original_hypotheses": train_df['text_2']
        },
        out_file
    )

with open('../data/vte/tweet/' + "train.pkl", mode="wb") as out_file:
    pickle.dump(
        {
            "image_names": train_df['image_1_path'],
            "labels": train_df['label_idx'],
            "original_premises": train_df['text_1'],
            "original_hypotheses": train_df['text_2']
        },
        out_file
    )

# test set
with open('../data/te/tweet/' + "test.pkl", mode="wb") as out_file:
    pickle.dump(
        {
            "labels": test_df['label_idx'],
            "original_premises": test_df['text_1'],
            "original_hypotheses": test_df['text_2']
        },
        out_file
    )

with open('../data/vte/tweet/' + "test.pkl", mode="wb") as out_file:
    pickle.dump(
        {
            "image_names": test_df['image_1_path'],
            "labels": test_df['label_idx'],
            "original_premises": test_df['text_1'],
            "original_hypotheses": test_df['text_2']
        },
        out_file
    )
exit(0)









# SNLI
# load the data
glove_filename = '../glove.840B.300d.txt'
dev_filename = '.data/snli/snli_1.0/snli_1.0_dev.jsonl' # '../vsnli/VSNLI_1.0_dev.tsv'
train_filename = '.data/snli/snli_1.0/snli_1.0_train.jsonl' # '../vsnli/VSNLI_1.0_train.tsv'
test_filename = '.data/snli/snli_1.0/snli_1.0_test.jsonl' # '../vsnli/VSNLI_1.0_test.tsv'
test_hard_filename = '.data/snli/snli_1.0/snli_1.0_test_hard.jsonl' # '../vsnli/VSNLI_1.0_test_hard.tsv'
test_lexical_filename = '.data/snli/snli_1.0/snli_1.0_test_lexical.jsonl'
max_vocab = 300000
embedding_size = 300
hidden_size = 512

# NOTE: Basic preprocessing: Glove, token2id related
print("-- Building vocabulary")
embeddings, token2id, id2token = load_glove(glove_filename, max_vocab, embedding_size)
label2id = {"neutral": 0, "entailment": 1, "contradiction": 2}
id2label = {v: k for k, v in label2id.items()}
num_tokens = len(token2id)
num_labels = len(label2id)
print("Number of tokens: {}".format(num_tokens))
print("Number of labels: {}".format(num_labels))

with open('metadata' + ".index", mode="wb") as out_file:
    pickle.dump(
        {
            "embeddings": embeddings,
            "token2id": token2id,
            "id2token": id2token,
            "label2id": label2id,
            "id2label": id2label
        },
        out_file
    )

with open('metadata' + ".index", mode="rb") as out_file:
    metadata = pickle.load(out_file)

token2id = metadata["token2id"] 
id2token = metadata["id2token"] 
label2id = metadata["label2id"] 
id2label = metadata["id2label"]
embeddings = torch.tensor(metadata["embeddings"], dtype=torch.float32)
print(f"Embeddings shape: {embeddings.shape}")     
num_labels = len(label2id)  
print(f"Number of labels: {num_labels}")     

# # NOTE: Creating TE dataset
dev_labels, dev_padded_premises, dev_padded_hypotheses, \
            dev_original_premises, dev_original_hypotheses = load_te_dataset(dev_filename, token2id, label2id)
train_labels, train_padded_premises, train_padded_hypotheses, \
            train_original_premises, train_original_hypotheses = load_te_dataset(train_filename, token2id, label2id)
test_labels, test_padded_premises, test_padded_hypotheses, \
            test_original_premises, test_original_hypotheses = load_te_dataset(test_filename, token2id, label2id)
test_hard_labels, test_hard_padded_premises, test_hard_padded_hypotheses, \
            test_hard_original_premises, test_hard_original_hypotheses = load_te_dataset(test_hard_filename, token2id, label2id)

# NOTE: Creating VTE dataset
dev_labels, dev_padded_premises, dev_padded_hypotheses, \
            dev_image_names, dev_original_premises, dev_original_hypotheses = load_vte_dataset(dev_filename, token2id, label2id)
train_labels, train_padded_premises, train_padded_hypotheses, \
            train_image_names, train_original_premises, train_original_hypotheses = load_vte_dataset(train_filename, token2id, label2id)
test_labels, test_padded_premises, test_padded_hypotheses, \
            test_image_names, test_original_premises, test_original_hypotheses = load_vte_dataset(test_filename, token2id, label2id)
test_hard_labels, test_hard_padded_premises, test_hard_padded_hypotheses, \
            test_hard_image_names, test_hard_original_premises, test_hard_original_hypotheses = load_vte_dataset(test_hard_filename, token2id, label2id)

# NOTE: Dumping TE dataset
with open('../data/te/' + "dev.pkl", mode="wb") as out_file:
    pickle.dump(
        {
            "labels": dev_labels,
            "padded_premises": dev_padded_premises,
            "padded_hypotheses": dev_padded_hypotheses,
            "original_premises": dev_original_premises,
            "original_hypotheses": dev_original_hypotheses
        },
        out_file
    )

with open('../data/te/' + "train.pkl", mode="wb") as out_file:
    pickle.dump(
        {
            "labels": train_labels,
            "padded_premises": train_padded_premises,
            "padded_hypotheses": train_padded_hypotheses,
            "original_premises": train_original_premises,
            "original_hypotheses": train_original_hypotheses
        },
        out_file
    )

with open('../data/te/' + "test.pkl", mode="wb") as out_file:
    pickle.dump(
        {
            "labels": test_labels,
            "padded_premises": test_padded_premises,
            "padded_hypotheses": test_padded_hypotheses,
            "original_premises": test_original_premises,
            "original_hypotheses": test_original_hypotheses
        },
        out_file
    )

with open('../data/te/' + "test_hard.pkl", mode="wb") as out_file:
    pickle.dump(
        {
            "labels": test_hard_labels,
            "padded_premises": test_hard_padded_premises,
            "padded_hypotheses": test_hard_padded_hypotheses,
            "original_premises": test_hard_original_premises,
            "original_hypotheses": test_hard_original_hypotheses
        },
        out_file
    )

# NOTE: Dumping VTE dataset 
with open('../data/vte/' + "dev.pkl", mode="wb") as out_file:
    pickle.dump(
        {
            "labels": dev_labels,
            "padded_premises": dev_padded_premises,
            "padded_hypotheses": dev_padded_hypotheses,
            "image_names": dev_image_names,
            "original_premises": dev_original_premises,
            "original_hypotheses": dev_original_hypotheses
        },
        out_file
    )

with open('../data/vte/' + "train.pkl", mode="wb") as out_file:
    pickle.dump(
        {
            "labels": train_labels,
            "padded_premises": train_padded_premises,
            "padded_hypotheses": train_padded_hypotheses,
            "image_names": train_image_names,
            "original_premises": train_original_premises,
            "original_hypotheses": train_original_hypotheses
        },
        out_file
    )

with open('../data/vte/' + "test.pkl", mode="wb") as out_file:
    pickle.dump(
        {
            "labels": test_labels,
            "padded_premises": test_padded_premises,
            "padded_hypotheses": test_padded_hypotheses,
            "image_names": test_image_names,
            "original_premises": test_original_premises,
            "original_hypotheses": test_original_hypotheses
        },
        out_file
    )

with open('../data/vte/' + "test_hard.pkl", mode="wb") as out_file:
    pickle.dump(
        {
            "labels": test_hard_labels,
            "padded_premises": test_hard_padded_premises,
            "padded_hypotheses": test_hard_padded_hypotheses,
            "image_names": test_hard_image_names,
            "original_premises": test_hard_original_premises,
            "original_hypotheses": test_hard_original_hypotheses
        },
        out_file
    )

# Preprocess test_lexical data
with open('../data/vte/' + "train.pkl", mode="rb") as out_file:
    train_vte = pickle.load(out_file)

premises = train_vte['original_premises']
hypothesis = train_vte['original_hypotheses']
image_names = train_vte['image_names']

premise_image_mapper, hypothesis_image_mapper = dict(), dict()
for i in range(len(premises)):
    premise_image_mapper[premises[i]] = image_names[i]
    hypothesis_image_mapper[hypothesis[i]] = image_names[i]

labels = []
padded_premises = []
padded_hypotheses = []
image_names = []
original_premises = []
original_hypotheses = []
lexical_data = []

with jsonlines.open(test_lexical_filename) as reader:
    not_found = 0
    for row in reader:
        label = row['gold_label'].strip()
        premise_tokens = row['sentence1'].strip().split()
        hypothesis_tokens = row['sentence2'].strip().split()
        premise = row['sentence1'].strip()
        hypothesis = row['sentence2'].strip()
        if premise in premise_image_mapper:
            image = premise_image_mapper[premise]
        elif premise in hypothesis_image_mapper:
            image = hypothesis_image_mapper[premise]
        else:
            not_found += 1
            
        labels.append(label2id[label])
        padded_premises.append([token2id.get(token, token2id["#unk#"]) for token in premise_tokens])
        padded_hypotheses.append([token2id.get(token, token2id["#unk#"]) for token in hypothesis_tokens])
        image_names.append(image)
        original_premises.append(premise)
        original_hypotheses.append(hypothesis)

    labels = np.array(labels)

with open('../data/vte/' + "test_lexical.pkl", mode="wb") as out_file:
    pickle.dump(
        {
            "labels": labels,
            "padded_premises": padded_premises,
            "padded_hypotheses": padded_hypotheses,
            "image_names": image_names,
            "original_premises": original_premises,
            "original_hypotheses": original_hypotheses
        },
        out_file
    )

with open('../data/te/' + "test_lexical.pkl", mode="wb") as out_file:
    pickle.dump(
        {
            "labels": labels,
            "padded_premises": padded_premises,
            "padded_hypotheses": padded_hypotheses,
            "original_premises": original_premises,
            "original_hypotheses": original_hypotheses
        },
        out_file
    )