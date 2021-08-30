#!/usr/bin/env python
# coding=utf-8
'''
Author: Yuxiang Yang
Date: 2021-08-19 00:32:07
LastEditors: Yuxiang Yang
LastEditTime: 2021-08-30 17:09:27
FilePath: /Chinese-NER/models/bilstm_for_ner.py
Description: 
'''
import torch.nn as nn
from torch.nn import LayerNorm
from .layers.crf import CRF
from torch.nn import CrossEntropyLoss
from losses.focal_loss import FocalLoss
from losses.label_smoothing import LabelSmoothingCrossEntropy

class SpatialDropout(nn.Dropout2d):
    def __init__(self, p=0.6):
        super(SpatialDropout, self).__init__(p=p)

    def forward(self, x):
        x = x.unsqueeze(2)  # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x


class BiLSTMForNer(nn.Module):
    def __init__(self, args):
        super(BiLSTMForNer, self).__init__()
        self.embedding_size = args.embedding_size
        self.model_type = args.model_type
        self.embedding = nn.Embedding(args.vocab_size, args.embedding_size)
        self.bilstm = nn.LSTM(input_size=args.embedding_size,
                              hidden_size=args.hidden_size,
                              num_layers=2,
                              batch_first=True,
                              dropout=args.drop_p,
                              bidirectional=True)
        self.dropout = SpatialDropout(args.drop_p)
        self.layer_norm = LayerNorm(args.hidden_size * 2)
        self.classifier = nn.Linear(args.hidden_size * 2, args.num_labels)
        self.use_crf = args.use_crf
        self.loss_type = args.loss_type
        self.num_labels = args.num_labels
        if args.use_crf:
            self.crf = CRF(num_tags=args.num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels, token_type_ids=None, input_lens=None):
        embs = self.embedding(input_ids)
        embs = self.dropout(embs)
        embs = embs * attention_mask.float().unsqueeze(2)
        sequence_output, _ = self.bilstm(embs)
        sequence_output= self.layer_norm(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            if self.use_crf:
                loss = self.crf(emissions=logits, tags=labels, mask=attention_mask)
                outputs =(-1*loss,)+outputs
            else:
                assert self.loss_type in ['ce', 'fl', 'lsc']
                if self.loss_type == 'ce':
                    loss_fct = CrossEntropyLoss(ignore_index=0)
                elif self.loss_type == 'fl':
                    loss_fct = FocalLoss(ignore_index=0)
                elif self.loss_type == 'lsc':
                    loss_fct = LabelSmoothingCrossEntropy(ignore_index=0)
                
                if attention_mask is not None:
                    active_loss = attention_mask.contiguous().view(-1) == 1
                    active_logits = logits.contiguous().view(-1, self.num_labels)[active_loss]
                    active_targets = labels.contiguous().view(-1)[active_loss]
                    loss = loss_fct(active_logits, active_targets)
                else:
                    loss = loss_fct(logits.contiguous().view(-1, self.num_labels), labels.contiguous().view(-1))
                outputs = (loss,) + outputs
        return outputs # (loss), scores





