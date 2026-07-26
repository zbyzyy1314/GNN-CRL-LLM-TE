"""
GNN for TE — Message Passing + Edge Features + Enhanced Node Features.
Simple O(E) per layer, fast training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MessagePassLayer(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim):
        super().__init__()
        self.W_msg = nn.Linear(in_dim + edge_dim, out_dim)
        self.W_self = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, edge_feat):
        B, N, _ = x.shape
        src_idx, dst_idx = edge_index[0], edge_index[1]
        E = edge_index.shape[1]
        src_h = x[:, src_idx]
        ef = edge_feat.unsqueeze(0).expand(B, -1, -1)
        msgs = F.gelu(self.W_msg(torch.cat([src_h, ef], dim=-1)))
        out = torch.zeros(B, N, msgs.shape[-1], device=x.device)
        out.scatter_add_(1, dst_idx.view(1, E, 1).expand(B, -1, msgs.shape[-1]), msgs)
        return out + self.W_self(x)


class MPBlock(nn.Module):
    def __init__(self, dim, edge_dim):
        super().__init__()
        self.mp = MessagePassLayer(dim, dim, edge_dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, ei, ef):
        return self.norm(F.gelu(x + self.mp(x, ei, ef)))


def _edge_data(topo):
    srcs, dsts, feats = [], [], []
    caps = topo.link_capacities; cmax = caps.max()
    for (s, d), lidx in topo.link_sd_to_idx.items():
        srcs.append(s); dsts.append(d)
        feats.append([
            np.log(1+caps[lidx])/np.log(1+cmax),
            topo.link_weights[lidx]/max(1,topo.link_weights.max()),
            np.log(1+topo.graph.out_degree(s))/np.log(1+max(1,topo.num_links)),
            np.log(1+topo.graph.in_degree(d))/np.log(1+max(1,topo.num_links)),
        ])
    return (torch.tensor([srcs,dsts],dtype=torch.long),
            torch.tensor(feats,dtype=torch.float32), len(feats[0]))


class TEGNNPolicy(nn.Module):
    """GNN policy network. Use TEGNNLLMPolicy for LLM-enhanced version."""
    def __init__(self, topo, hidden_dim=256, num_layers=3, max_k=8, top_k=3, temporal=False, history_len=12):
        super().__init__()
        self.num_nodes = topo.num_nodes
        self.num_pairs = topo.num_pairs
        self.max_k = max_k
        self.temporal = temporal
        sd = torch.tensor(topo.pair_idx_to_sd, dtype=torch.long)
        self.register_buffer('pair_src', sd[:,0])
        self.register_buffer('pair_dst', sd[:,1])
        ei, ef, ed = _edge_data(topo)
        self.register_buffer('edge_index', ei)
        self.register_buffer('edge_feat', ef)
        self.enc = nn.Sequential(nn.Linear(topo.num_nodes*2,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,hidden_dim))
        self.blocks = nn.ModuleList([MPBlock(hidden_dim,ed) for _ in range(num_layers)])
        self.dec = nn.Sequential(nn.Linear(hidden_dim*2+1,hidden_dim*2),nn.GELU(),nn.Linear(hidden_dim*2,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,max_k))
        if temporal:
            self.temp_enc = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
            )
        for m in self.modules():
            if isinstance(m,nn.Linear): nn.init.orthogonal_(m.weight,gain=0.5)
            if hasattr(m,'bias') and m.bias is not None: nn.init.constant_(m.bias,0)

    def _nf(self, tm):
        B,N=tm.shape[:2]
        v=torch.cat([tm, tm.transpose(1,2)], dim=-1)
        return v/v.max().clamp(min=1e-8)

    def encode_nodes(self, tm):
        """Extract node embeddings without decoder. Used by LLM-enhanced version."""
        ne = self.enc(self._nf(tm))
        for b in self.blocks:
            ne = b(ne, self.edge_index, self.edge_feat)
        return ne

    def decode_pairs(self, ne, tm, path_mask=None):
        """Decode node embeddings to path logits."""
        s,d=ne[:,self.pair_src],ne[:,self.pair_dst]
        logits=self.dec(torch.cat([s,d,tm[:,self.pair_src,self.pair_dst].unsqueeze(-1)],-1))
        if path_mask is not None:logits=torch.where(path_mask.unsqueeze(0),logits,torch.full_like(logits,-1e9))
        return logits

    def forward(self, tm, path_mask=None):
        if self.temporal:
            B, H, N, _ = tm.shape
            tm_flat = tm.reshape(B * H, N, N)
            ne = self.encode_nodes(tm_flat)  # (B*H, N, D)
            ne = ne.reshape(B, H, N, -1)
            # Temporal aggregation via Conv1d
            ne = ne.permute(0, 2, 3, 1)  # (B, N, D, H)
            ne = ne.reshape(B * N, -1, H)  # (B*N, D, H)
            ne = self.temp_enc(ne)  # (B*N, D, H) -> (B*N, D, H)
            ne = ne.mean(dim=-1)  # (B*N, D) pool over time
            ne = ne.reshape(B, N, -1)  # (B, N, D)
            tm_cur = tm[:, -1]  # last TM for demand
            return self.decode_pairs(ne, tm_cur, path_mask)
        ne=self.encode_nodes(tm)
        return self.decode_pairs(ne, tm, path_mask)

    def get_action(self,x,path_mask,deterministic=False):
        logits=self.forward(x,path_mask);probs=F.softmax(logits,-1)
        dist=torch.distributions.Categorical(probs)
        actions=probs.argmax(-1) if deterministic else dist.sample()
        return actions,dist.log_prob(actions),dist.entropy()
    def transfer_state_dict(self):
        """Extract topology-independent layers for cross-topology transfer."""
        state = {}
        for name, param in self.blocks.state_dict().items():
            state[f'blocks.{name}'] = param
        return state

    def load_transfer(self, state_dict):
        """Load topology-independent layers from another topology."""
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        return missing, unexpected


class TEGNNLLMPolicy(TEGNNPolicy):
    """GNN + LLM enhanced policy. GNN encodes structure, LLM adds semantic reasoning."""

    def __init__(self, topo, llm_encoder, hidden_dim=256, num_layers=3, max_k=8):
        super().__init__(topo, hidden_dim=hidden_dim, num_layers=num_layers, max_k=max_k)
        self.llm = llm_encoder

    def encode_nodes(self, tm):
        """GNN encode → LLM enhance → enhanced node embeddings."""
        ne = super().encode_nodes(tm)                # (B, N, D) from GNN
        enhanced = self.llm(tm, ne)                   # (B, N, D) LLM reasoned
        return enhanced

    def forward(self, tm, path_mask=None):
        ne = self.encode_nodes(tm)
        return self.decode_pairs(ne, tm, path_mask)


class GNNValueNetwork(nn.Module):
    def __init__(self, topo, hidden_dim=256, num_layers=3, top_k=3):
        super().__init__()
        self.num_nodes=topo.num_nodes
        sd=torch.tensor(topo.pair_idx_to_sd,dtype=torch.long)
        self.register_buffer('pair_src',sd[:,0]);self.register_buffer('pair_dst',sd[:,1])
        ei,ef,ed=_edge_data(topo)
        self.register_buffer('edge_index',ei);self.register_buffer('edge_feat',ef)
        self.enc=nn.Sequential(nn.Linear(topo.num_nodes*2,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,hidden_dim))
        self.blocks=nn.ModuleList([MPBlock(hidden_dim,ed) for _ in range(num_layers)])
        self.ro=nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,1))
        for m in self.modules():
            if isinstance(m,nn.Linear):nn.init.orthogonal_(m.weight,gain=1.0)
            if hasattr(m,'bias') and m.bias is not None:nn.init.constant_(m.bias,0)

    def forward(self,tm):
        B,N=tm.shape[:2]
        v=torch.cat([tm, tm.transpose(1,2)], dim=-1)
        ne=self.enc(v/v.max().clamp(min=1e-8))
        for b in self.blocks:ne=b(ne,self.edge_index,self.edge_feat)
        return self.ro(ne.mean(1)).squeeze(-1)
