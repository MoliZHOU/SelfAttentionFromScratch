import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedSelfAttetion(nn.Module):
    def __init__(self, d_model = 2,  # dimension of the model/number of word embedding values per token
                 row_dim=0, 
                 col_dim=1):
        super().__init__()
        self.col_dim = col_dim

        self.W_q = nn.Linear(in_features=d_model, out_features=d_model,bias=False)
        self.W_k= nn.Linear(in_features=d_model, out_features=d_model,bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model,bias=False)

    def forward(self, token_encodings, mask=None):
        ## Calculate the Masked Self-Attention values for each token
        q = self.W_q(token_encodings)
        k = self.W_k(token_encodings)
        v = self.W_v(token_encodings)
        
        sims= torch.matmul(q, k.transpose(0,1)) #交换第0维和第1维的内容
        # 除以根号d_k
        scale = sims / (k.size(self.col_dim) ** 0.5)
        if mask is not None:
            ## Here we are masking out things we don't want to pay attention to
            ##
            ## We replace values we wanted masked out
            ## with a very small negative number so that the SoftMax() function
            ## will give all masked elements an output value (or "probability") of 0.
            scaled_sims = scale.masked_fill(mask=mask, value=-1e9)
        
        attention_weights = F.softmax(scaled_sims, dim =self.col_dim)
        
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
    MaskSelfAttention = MaskedSelfAttetion(d_model=2,
                                row_dim=0,
                                col_dim=1)
    
    ## create the mask so that we don't use
    ## tokens that come after a token of interest
    mask = torch.tril(torch.ones(3, 3)) #取下三角为1
    mask = mask == 0 #转换为布尔遮罩
    mask # print out the mask

    ## calculate basic attention for the token encodings
    MaskSelfAttention(encodings_matrix, mask)

    # Verify

