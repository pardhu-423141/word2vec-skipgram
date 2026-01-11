import torch
import torch.nn as nn

class SkipGramNegSampling(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        self.out_embed = nn.Embedding(vocab_size, embed_dim)

        self.in_embed.weight.data.uniform_(-0.5/embed_dim, 0.5/embed_dim)
        self.out_embed.weight.data.uniform_(-0.5/embed_dim, 0.5/embed_dim)

    def forward(self, target, context, negatives):
        
        v = self.in_embed(target)            
        u_pos = self.out_embed(context)      
        u_neg = self.out_embed(negatives)    

        pos_score = torch.sum(v * u_pos, dim=1)
        pos_loss = torch.log(torch.sigmoid(pos_score))

        neg_score = torch.bmm(u_neg, v.unsqueeze(2)).squeeze()
        neg_loss = torch.sum(torch.log(torch.sigmoid(-neg_score)), dim=1)

        loss = -(pos_loss + neg_loss).mean()
        return loss
