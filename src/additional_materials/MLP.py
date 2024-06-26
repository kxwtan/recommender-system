import torch
import torch.nn as nn

class Multi_Layer_Perceptron(nn.Module):
    def __init__(self, args, num_users, num_items):
        super(Multi_Layer_Perceptron, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.factor_num = args.factor_num
        self.layers = args.layers

        # Embedding Layers
        self.embedding_user = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.factor_num)
        self.embedding_item = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.factor_num)

        # Fully connected layers (linear + ReLU) layers
        self.fc_layers = nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            self.fc_layers.append(nn.Linear(in_size, out_size))

        self.logits = nn.Linear(in_features=self.layers[-1], out_features=1)
        self.logistic = nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)

        # concatenate user and item embeddings
        vector = torch.cat([user_embedding, item_embedding], dim=-1)  # the concat latent vector

        # pass through a series of fully connected layers with ReLU Activation
        for idx, _ in enumerate(range(len(self.fc_layers))):
            vector = self.fc_layers[idx](vector)
            vector = nn.ReLU()(vector)
            vector = nn.BatchNorm1d()(vector)
            vector = nn.Dropout(p=0.5)(vector)
        logits = self.logits(vector)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass
