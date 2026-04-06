import torch
import torch.nn as nn
import math


class PatchEmbed1D(nn.Module):

  def __init__(self, patch_size, embed_dim, num_leads=12):
    super().__init__()
    self.num_leads = num_leads
    self.patch_size = patch_size
    self.embed_dim = embed_dim

    self.proj = nn.Conv1d(num_leads, num_leads*embed_dim, patch_size, groups=num_leads, stride=patch_size) # outputs [B, L*D, N] Where D = embed dimension and N is number of patches

  def forward(self,x):
    x = self.proj(x) # B L*D, N

    B, _, N = x.size()
    x = x.view(B, self.num_leads, self.embed_dim, N).transpose(2,3) # B, L, N, D
    return x

class PositionalEncoding(nn.Module):
  def __init__(self, embed_dim, max_seq_len):
    super().__init__()
    position_encoding = torch.zeros(max_seq_len, embed_dim) # N, D
    position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1) # N 1
    div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
    position_encoding[:, 0::2] = torch.sin(position * div_term) # every odd term
    position_encoding[:, 1::2] = torch.cos(position * div_term) # every even term
    position_encoding = position_encoding.unsqueeze(0).unsqueeze(0)
    self.register_buffer('position_encoding', position_encoding) # puts the pos encoding on gpu 1 N D

  def forward(self, x):
    # x: B L N D
    return x + self.position_encoding[:,:,:x.shape[-2]]

class LeadEncoding(nn.Module):
  def __init__(self, num_leads, embed_dim):
    super().__init__()

    lead_encoding = torch.zeros(num_leads, embed_dim) # N, D
    lead = torch.arange(0, num_leads, dtype=torch.float).unsqueeze(1) # N 1
    div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
    lead_encoding[:, 0::2] = torch.sin(lead * div_term) # every odd term
    lead_encoding[:, 1::2] = torch.cos(lead * div_term) # every even term
    lead_encoding = lead_encoding.unsqueeze(0).unsqueeze(2)
    self.register_buffer('lead_encoding', lead_encoding) # puts the pos encoding on gpu 1 N D

  def forward(self, x):


    return x + self.lead_encoding


class FeedForward(nn.Module):
  def __init__(self, embed_dim, mlp_hidden, dropout):
    super().__init__()
    self.mlp = nn.Sequential(
        #nn.LayerNorm(embed_dim), change
        nn.Linear(embed_dim, mlp_hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(mlp_hidden, embed_dim),
        nn.GELU(),
        nn.Dropout(dropout)
    )

  def forward(self, x):
    return x + self.mlp(x)

class TransformerBlock(nn.Module):
  def __init__(self, embed_dim, num_heads, mlp_hidden, dropout, device):
    super().__init__()

    self.num_heads = num_heads
    self.device = device

    self.s_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
    self.s_mlp = FeedForward(embed_dim, mlp_hidden, dropout)

    self.t_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
    self.t_mlp = FeedForward(embed_dim, mlp_hidden, dropout)


    self.t_norm = nn.LayerNorm(embed_dim)
    self.s_norm = nn.LayerNorm(embed_dim)

    self.normMLP = nn.LayerNorm(embed_dim) # change
    self.mlp = FeedForward(embed_dim, mlp_hidden, dropout) #change

    self.fuse = nn.Linear(2*embed_dim, embed_dim)



  def forward(self, x):

    # x: (B, C, N, D)
    B, C, N, D = x.shape


    # Random Masking
    # Temporal Branch with masking, across tokens per lead
    if self.training:
      t_mask = (torch.rand((N,N), device=self.device) > 0.7) # bool mask indicates which positions (true) are not allowed to attend.

    else:
      t_mask = None


    tc_mask = torch.triu(torch.full((N, N), float('-inf'), device=self.device), diagonal=1) # float masks have their values added to the corresponding positions. -inf functionally means no attention. Same as bool

    t = self.t_norm(x.view(B*C, N, D))
    t, _ = self.t_attention(t,t,t, attn_mask=t_mask, is_causal=False)


    t = self.t_mlp(t) #change

    t = t.view(B, C, N, D)

    # Spatial Branch with masking, across leads per token/patch

    if self.training:
      s_mask = (torch.rand((C,C), device=self.device) > 0.7) # prev 0.7

    else:
      s_mask = None


    sc_mask = torch.triu(torch.full((C, C), float('-inf'), device=self.device), diagonal=1)

    s = self.s_norm(x.transpose(1,2).contiguous().view(B*N, C, D))
    s, _ = self.s_attention(s,s,s, attn_mask=s_mask, is_causal=False)

    s = self.s_mlp(s) #change


    s = s.view(B, N, C, D).transpose(1,2).contiguous()

    fused = self.fuse(torch.cat([t, s], dim = -1)) # change

    x = x + fused


    x = self.mlp(self.normMLP(x)) #change
    """

    # No Masking


    t = self.t_norm(x.view(B*C, N, D))
    t, _ = self.t_attention(t,t,t)


    t = self.t_mlp(t) #change

    t = t.view(B, C, N, D)


    s = self.s_norm(x.transpose(1,2).contiguous().view(B*N, C, D))
    s, _ = self.s_attention(s,s,s,)

    s = self.s_mlp(s) #change


    s = s.view(B, N, C, D).transpose(1,2).contiguous()

    fused = self.fuse(torch.cat([t, s], dim = -1)) # change

    x = x + fused


    x = x + self.mlp(self.normMLP(t)) #change

    """



    return x




class TransformerECG(nn.Module):
  def __init__(self, num_leads, embed_dim, patch_size, num_heads, depth, mlp_dim, dropout, device, max_seq_len):
    super().__init__()

    self.patch_size = patch_size
    self.embed_dim = embed_dim
    self.num_leads = num_leads



    self.patch = PatchEmbed1D(self.patch_size, self.embed_dim, self.num_leads)

    self.blocks = nn.ModuleList([
        TransformerBlock(embed_dim, num_heads, mlp_dim, dropout, device) for _ in range(depth)
    ])


    self.pos_encoding = PositionalEncoding(self.embed_dim, max_seq_len)
    self.lead_encoding = LeadEncoding(self.num_leads, self.embed_dim)

    self.lin_norm = nn.LayerNorm(2*embed_dim)


    self.mlp_head = nn.Linear(2*embed_dim, 5)




  def forward(self, x):

    #encodings and embeddings
    x = self.patch(x)
    x = self.pos_encoding(x)
    x = self.lead_encoding(x)

    B , L , N , D = x.size()


    for block in self.blocks:
      x = block(x)


    #print(x.shape) Classify per lead

    B, L, N, D = x.shape
    tokens = x.reshape(B, L * N, D)
    mean_pool = tokens.mean(dim=1) # (B, D)
    max_pool, _ = tokens.max(dim=1) # (B, D)
    pooled = torch.cat([mean_pool, max_pool], dim=1) # (B, 2*D)
    logits = self.mlp_head(self.lin_norm(pooled)) # (B, num_classes) return logits

    return logits







