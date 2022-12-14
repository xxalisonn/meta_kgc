import torch
import torch.nn as nn
import numpy as np

class Embedding(nn.Module):
    def __init__(self, dataset, parameter):
        super(Embedding, self).__init__()
        self.device = parameter["device"]
        self.ent2id = dataset["ent2id"]
        self.es = parameter["embed_dim"]
        self.rum = parameter["rum"]
        num_ent = len(self.ent2id)
        self.embedding = nn.Embedding(num_ent, self.es)
        self.rum_embedding = None
        if parameter["data_form"] in ["Pre-Train", "In-Train"]:
            self.ent2emb = dataset["ent2emb"]
            self.embedding.weight.data.copy_(torch.from_numpy(self.ent2emb))
            if self.rum is True:
                self.rum_embedding = nn.Embedding(num_ent, self.es)
                self.rum_embedding.weight.data.copy_(torch.from_numpy(self.ent2emb))
                self.rum_embedding.requires_grad_(False)
            if not parameter["fine_tune"]:
                self.embedding.requires_grad_(False)
        elif parameter["data_form"] in ["Discard"]:
            nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, triples):
        # idx = [
        #     [[self.ent2id[t[0]], self.ent2id[t[2]]] for t in batch] for batch in triples
        # ]
        pair_num = len(max(triples, key=len))
        idx_ = np.zeros((len(triples),pair_num,2))
        for i in range(len(triples)):
            for j in range(len(triples[i])):
                idx_[i][j][0] = self.ent2id[triples[i][j][0]]
                idx_[i][j][1] = self.ent2id[triples[i][j][2]]

        idx = torch.LongTensor(idx_).to(self.device)
        if self.rum_embedding is None:
            return self.embedding(idx).unsqueeze(2)
        else:
            return torch.stack((self.rum_embedding(idx), self.embedding(idx)), dim=2)
