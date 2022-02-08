import csv
import jsonlines, json

import numpy as np

from utils import pad_sequences


def load_te_dataset(filename, token2id, label2id):
    labels = []
    padded_premises = []
    padded_hypotheses = []
    original_premises = []
    original_hypotheses = []

    # with open(filename) as in_file:
    #     reader = csv.reader(in_file, delimiter="\t")
    #     next(reader, None)
    with jsonlines.open(filename) as reader:
        for row in reader:
            # label = row[0].strip()
            # premise_tokens = row[1].strip().split()
            # hypothesis_tokens = row[2].strip().split()
            # premise = row[4].strip()
            # hypothesis = row[5].strip()

            label = row['gold_label'].strip()
            if label == '-':
                continue
            premise_tokens = row['sentence1'].strip().split()
            hypothesis_tokens = row['sentence2'].strip().split()
            premise = row['sentence1'].strip()
            hypothesis = row['sentence2'].strip()

            labels.append(label2id[label])
            padded_premises.append([token2id.get(token, token2id["#unk#"]) for token in premise_tokens]) # NOTE: premise for LSTM models
            padded_hypotheses.append([token2id.get(token, token2id["#unk#"]) for token in hypothesis_tokens]) # NOTE: hypothesis for LSTM models
            original_premises.append(premise) # NOTE: premise for BERT models
            original_hypotheses.append(hypothesis) # NOTE: hypothesis for BERT models

        # padded_premises = pad_sequences(padded_premises, padding="post", value=token2id["#pad#"], dtype=np.long)
        # padded_hypotheses = pad_sequences(padded_hypotheses, padding="post", value=token2id["#pad#"], dtype=np.long)
        labels = np.array(labels)

        return labels, padded_premises, padded_hypotheses, original_premises, original_hypotheses


def load_vte_dataset(nli_dataset_filename, token2id, label2id):
    labels = []
    padded_premises = []
    padded_hypotheses = []
    image_names = []
    original_premises = []
    original_hypotheses = []

    with jsonlines.open(nli_dataset_filename) as reader:
        # reader = csv.reader(in_file, delimiter="\t")
        # next(reader, None)

        for row in reader:
            # label = row[0].strip()
            # premise_tokens = row[1].strip().split()
            # hypothesis_tokens = row[2].strip().split()
            # premise = row[4].strip()
            # hypothesis = row[5].strip()

            label = row['gold_label'].strip()
            if label == '-':
                continue

            premise_tokens = row['sentence1'].strip().split()
            hypothesis_tokens = row['sentence2'].strip().split()
            premise = row['sentence1'].strip()
            hypothesis = row['sentence2'].strip()
            image = row['captionID'].strip().split("#")[0]
            if image.startswith('vg'):
                continue

            labels.append(label2id[label])
            padded_premises.append([token2id.get(token, token2id["#unk#"]) for token in premise_tokens])
            padded_hypotheses.append([token2id.get(token, token2id["#unk#"]) for token in hypothesis_tokens])
            image_names.append(image)
            original_premises.append(premise)
            original_hypotheses.append(hypothesis)

        # padded_premises = pad_sequences(padded_premises, padding="post", value=token2id["#pad#"], dtype=np.long)
        # padded_hypotheses = pad_sequences(padded_hypotheses, padding="post", value=token2id["#pad#"], dtype=np.long)
        labels = np.array(labels)

    return labels, padded_premises, padded_hypotheses, image_names, original_premises, original_hypotheses


class ImageReader:
    def __init__(self, img_names_filename, img_features_filename):
        self._img_names_filename = img_names_filename
        self._img_features_filename = img_features_filename

        with open(img_names_filename) as in_file:
            img_names = json.load(in_file)

        with open(img_features_filename, mode="rb") as in_file:
            img_features = np.load(in_file)

        self._img_names_features = {filename: features for filename, features in zip(img_names, img_features)}

    def get_features(self, images_names):
        return np.array([self._img_names_features[image_name] for image_name in images_names])
