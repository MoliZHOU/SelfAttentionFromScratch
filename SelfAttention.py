from networkx.generators import spectral_graph_forge
import torch
import torch.nn as nn # for Module and linear classes
import torch.nn.functional as F # for softmax and relu functions

class SelfAttention(nn.Module):

    def __init__(self,d_model = 2,  # the number of embedding value for each token
            row_dim = 0,  # index of the row vector in the input data matrix
            col_dim=1): # index of the col vector in the input data matrix
        super().__init__()
        
        # untrained weight matrices
        self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        # in_features is how many rows of the Weight matrix
        # out_features is how many columns of the Weight matrix
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        
        self.row_dim = row_dim
        self.col_dim = col_dim

    def forward(self, token_encodings):
        ## Create the query, key and values using the encoding numbers
        ## associated with each token (token encodings)
        q = self.W_q(token_encodings)
        k = self.W_k(token_encodings)
        v = self.W_v(token_encodings)

        ## Compute similarities scores: (q * k^T)
        sims = torch.matmul(q, k.transpose(dim0=self.row_dim, dim1=self.col_dim))
        # not using k.T considering the batch dimension
        # 
        
        # scale the similarities
        scaled_sims = sims / (k.size(self.col_dim) ** 0.5) 

        #apply softmax to calculate the percentage of each token's value
        attention_weights = F.softmax(scaled_sims, dim=self.col_dim)
        
        attention_score = torch.matmul(attention_weights, v)
        return attention_score

if __name__ == "__main__":

    #example data
    ## create a matrix of token encodings...
    encodings_matrix = torch.tensor([[1.16, 0.23],
                                    [0.57, 1.36],
                                    [4.41, -2.16]])

    ## set the seed for the random number generator
    torch.manual_seed(42)

    ## create a basic self-attention ojbect
    selfAttention = SelfAttention(d_model=2,
                                row_dim=0,
                                col_dim=1)

    ## calculate basic attention for the token encodings
    selfAttention(encodings_matrix)

    