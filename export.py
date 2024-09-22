from modulefinder import Module

from model import UIE
import torch
from torch import nn

import pnnx

model = UIE.from_pretrained('uie-nano-pytorch')


class EmbeddingRepack(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.word_embeddings = model.encoder.embeddings.word_embeddings
        self.position_embeddings = model.encoder.embeddings.position_embeddings
        self.token_type_embeddings = model.encoder.embeddings.token_type_embeddings
        self.task_type_embeddings = model.encoder.embeddings.task_type_embeddings
        self.layer_norm = model.encoder.embeddings.LayerNorm

    def forward(self, input_ids, position_ids, token_type_ids, task_type_ids):
        inputs_embeddings = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(position_ids)
        task_type_embeddings = self.task_type_embeddings(task_type_ids)

        embeddings = inputs_embeddings + token_type_embeddings + position_embeddings + task_type_embeddings
        embeddings = self.layer_norm(embeddings)

        return embeddings


class EncoderRepack(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder.encoder
        self.pooler = model.encoder.pooler

    def forward(self, embeddings):
        encoder_output = self.encoder(embeddings)
        return encoder_output.last_hidden_state


class UIERepack(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.embedding_repack = EmbeddingRepack(model)
        self.encoder_repack = EncoderRepack(model)

        self.linear_start = model.linear_start
        self.linear_end = model.linear_end
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, position_ids, token_type_ids, task_type_ids):
        embeds = self.embedding_repack(input_ids, position_ids, token_type_ids,
                                       task_type_ids)
        sequence_output = self.encoder_repack(embeds)

        start_logits = self.linear_start(sequence_output)
        start_logits = torch.squeeze(start_logits, -1)
        start_prob = self.sigmoid(start_logits)

        end_logits = self.linear_end(sequence_output)
        end_logits = torch.squeeze(end_logits, -1)
        end_prob = self.sigmoid(end_logits)

        return start_prob, end_prob


model_repack = UIERepack(model)
model_repack.eval()

x_0 = torch.tensor([[1, 36, 143, 2, 249, 136, 585, 139, 28, 1598,
                     252, 560, 1296, 1038, 32, 67, 190, 220, 1311, 1097,
                     291, 85, 19, 1498, 357, 448, 628, 12, 12, 20,
                     352, 247, 1146, 329, 1853, 22, 4734, 42, 1266, 59,
                     349, 116, 192, 664, 12044, 2]], dtype=torch.long)
x_1 = torch.tensor([[i for i in range(x_0.size(1))]], dtype=torch.long)
x_2 = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
x_3 = torch.tensor([[0] * x_0.size(1)])

pnnx.export(model_repack, 'uie_nano_pnnx.pt', [x_0, x_1, x_2, x_3],
            inputs2=[torch.tensor([[1] * 16]), torch.tensor([[1] * 16]), torch.tensor([[1] * 16]),
                     torch.tensor([[1] * 16])])
