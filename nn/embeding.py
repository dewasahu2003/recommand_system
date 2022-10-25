import torch
import torch.nn as NN

n_embedding,dim=10,4

emb_1=NN.Embedding(n_embedding,dim)
print(emb_1.weight)
print(emb_1(torch.tensor([1,2,3])))

print('=================================')

emb_2=NN.Embedding(n_embedding,dim,padding_idx=5)
print(emb_2.weight)
print(emb_2(torch.tensor([1,2,3])).mean().backward())
print(emb_2.weight.grad)


print('============================')

emb_3=NN.Embedding(n_embedding,dim,norm_type=2,max_norm=1)
print(emb_3.weight.norm(dim=-1))