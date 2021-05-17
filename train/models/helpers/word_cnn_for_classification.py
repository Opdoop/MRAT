"""
Word CNN for Classification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

"""


import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from train.models.tokenizers import GloveTokenizer
from train.models.helpers import GloveEmbeddingLayer
from train.models.helpers.utils import load_cached_state_dict
from train.shared import utils


class WordCNNForClassification(nn.Module):
    """A convolutional neural network for text classification.

    We use different versions of this network to pretrain models for
    text classification.
    """

    def __init__(
        self,
        hidden_size=150,
        dropout=0.3,
        num_labels=2,
        max_seq_length=128,
        model_path=None,
        emb_layer_trainable=True,
    ):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.emb_layer = GloveEmbeddingLayer(emb_layer_trainable=emb_layer_trainable)
        self.word2id = self.emb_layer.word2id
        self.encoder = CNNTextLayer(
            self.emb_layer.n_d, widths=[3, 4, 5], filters=hidden_size
        )
        d_out = 3 * hidden_size
        self.out = nn.Linear(d_out, num_labels)
        self.tokenizer = GloveTokenizer(
            word_id_map=self.word2id,
            unk_token_id=self.emb_layer.oovid,
            pad_token_id=self.emb_layer.padid,
            max_length=max_seq_length,
        )

        if model_path is not None:
            self.load_from_disk(model_path)

    def load_from_disk(self, model_path):
        self.load_state_dict(load_cached_state_dict(model_path))
        self.to(utils.device)
        self.eval()

    def forward(self, _input, perturbed_input=None, mix_ratio=None, feature_mixup=True):

        if perturbed_input is None:
            emb = self.emb_layer(_input)
            emb = self.drop(emb)

            output = self.encoder(emb)

            output = self.drop(output)
            pred = self.out(output)
        else:
            if feature_mixup:
                emb_noraml = self.emb_layer(_input)
                emb_perturb = self.emb_layer(perturbed_input)
                m = mix_ratio #np.float32(np.random.beta(1, 1))
                emb_mixup = (1 - m) * emb_perturb + m * emb_noraml

                emb = self.drop(emb_mixup)

                output = self.encoder(emb)

                output = self.drop(output)
                pred = self.out(output)
            else:
                emb_noraml = self.emb_layer(_input)
                emb_perturb = self.emb_layer(perturbed_input)

                emb_noraml = self.drop(emb_noraml)
                emb_perturb = self.drop(emb_perturb)

                output_normal = self.encoder(emb_noraml)
                output_perturb = self.encoder(emb_perturb)

                m = mix_ratio #np.float32(np.random.beta(1, 1))
                output = (1 - m) * output_perturb + m * output_normal
                output = self.drop(output)
                pred = self.out(output)
        return pred

    def get_input_embeddings(self):
        return self.emb_layer.embedding


class CNNTextLayer(nn.Module):
    def __init__(self, n_in, widths=[3, 4, 5], filters=100):
        super().__init__()
        Ci = 1
        Co = filters
        h = n_in
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (w, h)) for w in widths])

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, Ci, len, d)
        x = [
            F.relu(conv(x)).squeeze(3) for conv in self.convs1
        ]  # [(batch, Co, len), ...]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,Co), ...]
        x = torch.cat(x, 1)
        return x
