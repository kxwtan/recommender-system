import torch
import torch.nn as nn

class NeuMF(nn.Module):
    
    def __init__(self, args, user_count, item_count):
        super(NeuMF, self).__init__()

        self.user_count = user_count
        self.item_count = item_count

        self.latent_dim_mf = args.factor_num
        self.latent_dim_mlp = int(args.layers[0]/2)

        # Matrix Factorization embedding layers
        self.embedding_user_mf = nn.Embedding(num_embeddings=self.user_count, embedding_dim=self.latent_dim_mf)
        self.embedding_item_mf = nn.Embedding(num_embeddings=self.item_count, embedding_dim=self.latent_dim_mf)
        
        # MLP embedding layers
        self.embedding_user_mlp = nn.Embedding(num_embeddings=self.user_count, embedding_dim=self.latent_dim_mlp)
        self.embedding_item_mlp = nn.Embedding(num_embeddings=self.item_count, embedding_dim=self.latent_dim_mlp)

        # initialize the MLP layers
        self.fc_layers = nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(args.layers[:-1], args.layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
            self.fc_layers.append(nn.ReLU())

        self.logits = nn.Linear(in_features=args.layers[-1] + self.latent_dim_mf, out_features=1)
        self.logistic = nn.Sigmoid()
        self.init_weight()

    def init_weight(self):

        # initialize embedding layers with small random values (mean = 0, std = 0.01)
        nn.init.normal_(self.embedding_user_mlp.weight, std=0.01)
        nn.init.normal_(self.embedding_item_mlp.weight, std=0.01)
        nn.init.normal_(self.embedding_user_mf.weight, std=0.01)
        nn.init.normal_(self.embedding_item_mf.weight, std=0.01)
        
        # Use xavier_uniform to initialize the weights of the linear layers
        # maintains proper variance of weights across layers
        for m in self.fc_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                
        nn.init.xavier_uniform_(self.logits.weight)

        # Set the bias of linear layers to zero ensuring that the intial predictions are unbiased
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, user_indices, item_indices):
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)

        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)
        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)

        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)

        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.logits(vector)
        rating = self.logistic(logits)
        return rating.squeeze()