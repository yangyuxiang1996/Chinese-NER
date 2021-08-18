import torch
import torch.nn as nn
from torch.nn import LayerNorm
from .layers.crf import CRF
from .transformers.modeling_bert import BertModel

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
        if args.model_type == 'bert_bilstm_crf':
            assert args.model_name_or_path != ""
            self.embedding = BertModel.from_pretrained(args.model_name_or_path)
        else:
            self.embedding = nn.Embedding(args.vocab_size, args.embedding_size)
        self.bilstm = nn.LSTM(input_size=args.embedding_size,
                              hidden_size=args.hidden_size,
                              num_layers=2,
                              batch_first=True,
                              dropout=args.drop_p,
                              bidirectional=True)
        self.dropout = SpatialDropout(args.drop_p)
        self.layer_norm = LayerNorm(args.hidden_size * 2)
        self.classifier = nn.Linear(args.hidden_size*2, args.num_labels)
        self.crf = CRF(num_tags=args.num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels, token_type_ids=None):
        if self.model_type == 'bert_bilstm_crf':
            embs = self.embedding(input_ids)[0]
        else:
            embs = self.embedding(input_ids)
        embs = self.dropout(embs)
        embs = embs * attention_mask.float().unsqueeze(2)
        seqence_output, _ = self.bilstm(embs)
        seqence_output= self.layer_norm(seqence_output)
        logits = self.classifier(seqence_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf(emissions = logits, tags=labels, mask=attention_mask)
            outputs =(-1*loss,)+outputs
        return outputs # (loss), scores





