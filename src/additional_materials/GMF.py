import torch
import torch.nn as nn

class Generalized_Matrix_Factorization(nn.Module):
    def __init__(self, args, num_users, num_items):
        super(Generalized_Matrix_Factorization, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.factor_num = args.factor_num

        # Embedding layers
        self.embedding_user = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.factor_num)
        self.embedding_item = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.factor_num)

        # Linear layer to combine user and item embeddings
        self.logits = nn.Linear(in_features=self.factor_num, out_features=1)

        # normalize rating
        self.logistic = nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        # embed user and item indices
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)

        # compute product of user and item embeddings
        element_product = torch.mul(user_embedding, item_embedding)
        logits = self.logits(element_product)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass
