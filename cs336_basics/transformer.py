import torch
from einops import rearrange, einsum
import math
import sys
import os
from collections.abc import Iterable,Callable
from typing import IO, Any, BinaryIO, Optional
import numpy as np
import numpy.typing as npt
from jaxtyping import Bool, Float, Int
from torch import Tensor
sys.path.insert(0, "/home/zhang/projects/cs336/cs336-assignment1")
from cs336_basics.bpe import tokenize,Tokenizer

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.W=torch.nn.Parameter(torch.randn(out_features,in_features,dtype=dtype,device=device))
        torch.nn.init.trunc_normal_(self.W, mean=0.0,std=math.sqrt(2 / (in_features + out_features)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.W,x,"d_out d_in,... d_in -> ... d_out")
    
class Embedding(torch.nn.Module):
    '''
    num_embeddings:vocab_size
    embedding_dim:d_model
    embedding用于将token_id通过查embedding_matrix转为一个d_model维的向量，用于以后计算
    '''
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.embedding=torch.nn.Parameter(torch.randn(num_embeddings,embedding_dim,device=device,dtype=dtype))
        torch.nn.init.trunc_normal_(self.embedding, mean=0.0,std=math.sqrt(2 / (num_embeddings + embedding_dim)))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding[token_ids]
    
class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.g=torch.nn.Parameter(torch.randn(d_model,device=device,dtype=dtype))
        torch.nn.init.trunc_normal_(self.g, mean=0.0,std=math.sqrt(2 / (1+d_model)))
        self.eps=eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms=(x.square().mean(dim=-1, keepdim=True) + self.eps).sqrt()
        result = (x / rms) * self.g.to(torch.float32)
        return result.to(in_dtype)
    

def SiLU(x:torch.Tensor):
    return x*torch.sigmoid(x)


class SwiGLU(torch.nn.Module):
    '''
    d_ff be set to approximately 8*d_model/3 in implementation
    '''
    def __init__(self, d_model:int, d_ff:int, device=None, dtype=None):
        super().__init__()
        self.W1=torch.nn.Parameter(torch.randn(d_ff,d_model,dtype=dtype,device=device))
        self.W2=torch.nn.Parameter(torch.randn(d_model,d_ff,dtype=dtype,device=device))
        self.W3=torch.nn.Parameter(torch.randn(d_ff,d_model,dtype=dtype,device=device))
        torch.nn.init.trunc_normal_(self.W1, mean=0.0,std=math.sqrt(2 / (d_ff+d_model)))
        torch.nn.init.trunc_normal_(self.W2, mean=0.0,std=math.sqrt(2 / (d_ff+d_model)))
        torch.nn.init.trunc_normal_(self.W3, mean=0.0,std=math.sqrt(2 / (d_ff+d_model)))
    
    def forward(self, x:torch.Tensor):
        W1x=einsum(self.W1,x,"d_ff d_model,... d_model->... d_ff")
        W3x=einsum(self.W3,x,"d_ff d_model,... d_model->... d_ff")
        new_x=SiLU(W1x)*W3x
        return einsum(new_x,self.W2,"... d_ff,d_model d_ff->... d_model")
    
import torch

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        
        # 优化点 1：将 inv_freq 注册为 buffer，这样它会随模型移动到 GPU，且不会被视作训练参数
        k = torch.arange(0, self.d_k, 2, device=self.device).float()
        inv_freq = 1.0 / (self.theta ** (k / self.d_k))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor|None=None) -> torch.Tensor:
        # x 形状通常为 [Batch, Seq, d_k]
        # token_positions 形状为 [Batch, Seq]
        
        # 1. 计算角度 angles: [Batch, Seq, d_k//2]
        # 这里需要确保 inv_freq 的维度能和 token_positions 匹配
        # token_positions: (B, S) -> (B, S, 1)
        # inv_freq: (d_k//2) -> (1, 1, d_k//2)
        device = x.device
        dtype = x.dtype
        if token_positions is None:
            # x 的形状假设为 [Batch, Num_Heads, Seq_Len, Head_Dim]
            seq_len = x.shape[-2] 
            # 生成 [0, 1, 2, ..., seq_len-1] 并扩展到与 Batch 一致
            token_positions = torch.arange(seq_len, device=device).unsqueeze(0).to(device)
        angles = token_positions.unsqueeze(-1) * self.inv_freq.reshape(1, 1, -1)
        angles = angles.to(dtype)

        # 2. 生成 cos 和 sin
        cos_emb = angles.cos()  # [B, S, d_k//2]
        sin_emb = angles.sin()  # [B, S, d_k//2]

        # 3. 应用 RoPE 旋转
        x1 = x[..., 0::2]   # 偶数索引元素
        x2 = x[..., 1::2]   # 奇数索引元素

        # 优化点 2：RoPE 的标准做法是 [x1*cos - x2*sin, x1*sin + x2*cos]
        # 但为了保持维度顺序，我们通常采用交替合并或 chunk 合并
        # 这里采用 stack 再 flatten 的方式来恢复 [..., d_k] 的交替顺序
        o1 = x1 * cos_emb - x2 * sin_emb
        o2 = x1 * sin_emb + x2 * cos_emb
        
        # 优化点 3：组合输出。如果直接 cat，维度会变成 [左边全是o1, 右边全是o2]
        # 我们需要交错排列以匹配原始 x 的 [0,1,2,3...] 结构
        out = torch.stack([o1, o2], dim=-1).flatten(-2)

        return out
    
def softmax(x:torch.Tensor,i:int):
    max_value,_=x.max(dim=i,keepdim=True)
    sub_value=x-max_value
    exp_value=torch.exp(sub_value)
    sum_exp=exp_value.sum(dim=i,keepdim=True)
    return torch.where(sum_exp > 0, exp_value / sum_exp, torch.zeros_like(exp_value))

def scaled_dot_product_attention(Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,):
    QK=einsum(Q,K,'... queries d_k,... keys d_k->... queries keys')
    QK=QK/math.sqrt(Q.shape[-1])
    if mask is not None:
        QK = QK.masked_fill(mask == False, float('-inf'))
    W = torch.softmax(QK, dim=-1)
    return einsum(W,V,'... queries keys,... keys d_v->... queries d_v')

def multihead_self_attention(d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " hdk d_model"],
    k_proj_weight: Float[Tensor, " hdk d_model"],
    v_proj_weight: Float[Tensor, " hdv d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],) -> Float[Tensor, " ... sequence_length d_model"]:
    Q=einsum(q_proj_weight,in_features,'hdk d_model,... sequence_length d_model->... sequence_length hdk')
    K=einsum(k_proj_weight,in_features,'hdk d_model,... sequence_length d_model->... sequence_length hdk')
    V=einsum(v_proj_weight,in_features,'hdv d_model,... sequence_length d_model->... sequence_length hdv')
    Q=rearrange(Q,'... seq (h dk)->... h seq dk',h=num_heads)
    K=rearrange(K,'... seq (h dk)->... h seq dk',h=num_heads)
    V=rearrange(V,'... seq (h dv)->... h seq dv',h=num_heads)
    seq_len = in_features.shape[-2]
    mask = ~torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    X=rearrange(scaled_dot_product_attention(Q,K,V,mask=mask),'... h seq dv->... seq (h dv)')
    return einsum(o_proj_weight,X,'d hdv,... seq hdv->... seq d')

def multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    rpe=RotaryPositionalEmbedding(theta,d_model/num_heads,max_seq_len,device=in_features.device)
    Q=einsum(q_proj_weight,in_features,'hdk d_model,... sequence_length d_model->... sequence_length hdk')
    K=einsum(k_proj_weight,in_features,'hdk d_model,... sequence_length d_model->... sequence_length hdk')
    V=einsum(v_proj_weight,in_features,'hdv d_model,... sequence_length d_model->... sequence_length hdv')
    Q=rearrange(Q,'... seq (h dk)->... h seq dk',h=num_heads)
    K=rearrange(K,'... seq (h dk)->... h seq dk',h=num_heads)
    V=rearrange(V,'... seq (h dv)->... h seq dv',h=num_heads)
    Q=rpe.forward(Q,token_positions)
    K=rpe.forward(K,token_positions)
    seq_len = in_features.shape[-2]
    mask = ~torch.triu(torch.ones(seq_len, seq_len,device=in_features.device), diagonal=1).bool()
    X=rearrange(scaled_dot_product_attention(Q,K,V,mask=mask),'... h seq dv->... seq (h dv)')
    return einsum(o_proj_weight,X,'d hdv,... seq hdv->... seq d')

class TransformerBlock(torch.nn.Module):
    '''
            weights (dict[str, Tensor]):
            State dict of our reference implementation.
            The keys of this dictionary are:
            - `attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (d_model, d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is (d_model, d_model).
            - `ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
    '''
    def __init__(self, d_model,
        num_heads,
        d_ff,
        max_seq_len: int=1024,
        theta: float=10000,
        weights: dict[str, Tensor] | None = None,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn=SwiGLU(d_model,d_ff,device=device,dtype=dtype)
        self.q_proj = torch.nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.k_proj = torch.nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.v_proj = torch.nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))
        self.output_proj = torch.nn.Parameter(torch.empty(d_model, d_model, device=device, dtype=dtype))

        if weights is not None:
            self.load_state_dict_custom(weights)
        else:
            for p in self.parameters():
                if p.dim() > 1:
                    torch.nn.init.xavier_uniform_(p)
                else:
                    torch.nn.init.ones_(p)

    def load_state_dict_custom(self, weights: dict[str, Tensor]):
        with torch.no_grad():
            self.ln1.g.copy_(weights['ln1.weight'])
            self.ln2.g.copy_(weights['ln2.weight'])
            self.q_proj.copy_(weights['attn.q_proj.weight'])
            self.k_proj.copy_(weights['attn.k_proj.weight'])
            self.v_proj.copy_(weights['attn.v_proj.weight'])
            self.output_proj.copy_(weights['attn.output_proj.weight'])
            self.ffn.W1.copy_(weights['ffn.w1.weight'])
            self.ffn.W2.copy_(weights['ffn.w2.weight'])
            self.ffn.W3.copy_(weights['ffn.w3.weight'])

    def forward(self,in_features):
        i1=self.ln1.forward(in_features)
        i2=multihead_self_attention_with_rope(self.d_model,self.num_heads,self.max_seq_len,self.theta,self.q_proj,self.k_proj,self.v_proj,self.output_proj,i1)
        i3=i2+in_features
        i4=self.ln2.forward(i3)
        i5=self.ffn.forward(i4)
        return i5+i3
    
class Transformer(torch.nn.Module):
    '''
    Args:
        vocab_size (int): The number of unique items in the output vocabulary to be predicted.
        context_length (int): The maximum number of tokens to process at once.
        d_model (int): The dimensionality of the model embeddings and sublayer outputs.
        num_layers (int): The number of Transformer layers to use.
        num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
        rope_theta (float): The RoPE $\\Theta$ parameter.
        weights (dict[str, Tensor]):
            State dict of our reference implementation. {num_layers} refers to an
            integer between `0` and `num_layers - 1` (the layer index).
            The keys of this dictionary are:
            - `token_embeddings.weight`
                Token embedding matrix. Shape is (vocab_size, d_model).
            - `layers.{num_layers}.attn.q_proj.weight`
                The query projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.k_proj.weight`
                The key projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_k),
                so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.v_proj.weight`
                The value projections for all `num_heads` attention heads.
                Shape is (num_heads * (d_model / num_heads), d_model).
                The rows are ordered by matrices of shape (num_heads, d_v),
                so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
            - `layers.{num_layers}.attn.output_proj.weight`
                Weight of the multi-head self-attention output projection
                Shape is ((d_model / num_heads) * num_heads, d_model).
            - `layers.{num_layers}.ln1.weight`
                Weights of affine transform for the first RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `layers.{num_layers}.ffn.w1.weight`
                Weight of the first linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ffn.w2.weight`
                Weight of the second linear transformation in the FFN.
                Shape is (d_model, d_ff).
            - `layers.{num_layers}.ffn.w3.weight`
                Weight of the third linear transformation in the FFN.
                Shape is (d_ff, d_model).
            - `layers.{num_layers}.ln2.weight`
                Weights of affine transform for the second RMSNorm
                applied in the transformer block.
                Shape is (d_model,).
            - `ln_final.weight`
                Weights of affine transform for RMSNorm applied to the output of the final transformer block.
                Shape is (d_model, ).
            - `lm_head.weight`
                Weights of the language model output embedding.
                Shape is (vocab_size, d_model).
        in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
            `sequence_length` is at most `context_length`.
    '''
    def __init__(self,vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    device=None,
    dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        
        self.layers = torch.nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                weights=None,      # 统一在后面加载权重
                device=device,
                dtype=dtype
            )
            for _ in range(num_layers)
        ])
        
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
        
        if weights is not None:
            self.load_from_dict(weights)
        else:
            self._apply_default_init()

    def _apply_default_init(self):
        """如果没给权重，对所有参数进行基础初始化"""
        for name, p in self.named_parameters():
            if 'weight' in name or 'W' in name or 'embedding' in name:
                if p.dim() > 1:
                    torch.nn.init.trunc_normal_(p, std=0.02)
            elif 'g' in name: # RMSNorm
                torch.nn.init.ones_(p)

    def load_from_dict(self, weights: dict[str, Tensor]):
        """根据字典 key 加载权重"""
        with torch.no_grad():
            if 'token_embeddings.weight' in weights:
                self.token_embeddings.embedding.copy_(weights['token_embeddings.weight'])
            
            for i in range(self.num_layers):
                prefix = f"layers.{i}."
                layer_weights = {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}
                if layer_weights:
                    self.layers[i].load_state_dict_custom(layer_weights)
            
            if 'ln_final.weight' in weights:
                self.ln_final.g.copy_(weights['ln_final.weight'])
            if 'lm_head.weight' in weights:
                self.lm_head.W.copy_(weights['lm_head.weight'])

    def forward(self,in_indices):
        x=self.token_embeddings.forward(in_indices)
        for i in self.layers:
            x=i.forward(x)
        x=self.ln_final.forward(x)
        x=self.lm_head.forward(x)
        return x

def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    max_value,_=inputs.max(dim=-1,keepdim=True)
    inputs=inputs-max_value
    expsum=torch.exp(inputs).sum(dim=-1)
    result=torch.log(expsum)-inputs[torch.arange(inputs.shape[0],device=inputs.device), targets]
    return result.mean()


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or 0.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self,params,lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        if lr < 0.0: raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0: raise ValueError(f"Invalid beta1: {betas[0]}")
        if weight_decay < 0.0: raise ValueError(f"Invalid weight_decay: {weight_decay}")
        defaults = {'lr':lr, 'betas':betas, 'eps':eps, 'weight_decay':weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self,closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.
            weight_dacay=group['weight_decay']
            eps=group['eps']
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 1)  # Get iteration number from the state, or 0.
                v=state.get('v',torch.zeros_like(p))
                m=state.get('m',torch.zeros_like(p))
                beta1=group['betas'][0]
                beta2=group['betas'][1]
                lrt=lr*math.sqrt(1-math.pow(beta2,t))/(1-math.pow(beta1,t))
                grad = p.grad.data  
                p.data.mul_(1 - lr*weight_dacay) #p.data -= lr*weight_dacay*p.data的原位计算
                m.mul_(beta1).add_(grad, alpha=1 - beta1) #m=beta1*m+(1-beta1)*grad
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2) #v=beta2*v+(1-beta2)*grad*grad
                denom = v.sqrt().add_(eps) #p.data-=lrt*m/(torch.sqrt(v)+eps)
                p.data.addcdiv_(m, denom, value=-lrt)
                state["t"] = t + 1
                state['m']=m
                state['v']=v
        return loss
    
def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it<warmup_iters:
        return it/warmup_iters*max_learning_rate
    else:
        if it<=cosine_cycle_iters:
            return min_learning_rate+0.5*(1+math.cos((it-warmup_iters)*math.pi/(cosine_cycle_iters-warmup_iters)))*(max_learning_rate-min_learning_rate)
        else:
            return min_learning_rate
        
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return
    device = params_with_grad[0].grad.device
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2).to(device) for p in params_with_grad]), 2)
    clip_coeff = max_l2_norm / (total_norm + 1e-6)
    if clip_coeff < 1.0:
        for p in params_with_grad:
            p.grad.detach().mul_(clip_coeff)

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(dataset) - context_length, (batch_size,))
    x_list = [torch.from_numpy(dataset[i:i+context_length]).long() for i in ix]
    y_list = [torch.from_numpy(dataset[i+1:i+1+context_length]).long() for i in ix]
    X = torch.stack(x_list)
    Y = torch.stack(y_list)
    return (X.to(device),Y.to(device))

def save_checkpoint(model, optimizer, iteration, out):
    checkpoint={'iteration':iteration,
                'model':model.state_dict(),
                'optimizer':optimizer.state_dict()}
    torch.save(checkpoint,out)
    return

def load_checkpoint(src, model, optimizer):
    checkpoint=torch.load(src)
    state_dict = checkpoint['model']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    
    for k, v in state_dict.items():
        # 如果键名以 'module.' 开头，则截取掉这 7 个字符
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['iteration']

def train(vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    out,
    batch_size,
    iteration_step,
    data_source,
    warmup_iters,
    cosine_cycle_iters,
    device=None,
    dtype=None,
    rope_theta=10000,
    lr=1e-3, 
    min_learning_rate=1e-4, #将lr当作max_learning_rate
    betas=(0.9, 0.999), 
    eps=1e-8, 
    weight_decay=1e-2,
    max_norm=1,
    save_every=None,
    ):
    model=Transformer(vocab_size,context_length,d_model,num_layers,num_heads,d_ff,rope_theta,weights=None,device=device,dtype=dtype)
    model.to(device)
    optimizer=AdamW(model.parameters(),lr,betas,eps,weight_decay)
    data=np.memmap(data_source,dtype=np.uint16,mode='r')
    for i in range(iteration_step):
        # 1. 计算当前步对应的学习率
        current_lr = get_lr_cosine_schedule(
            it=i, 
            max_learning_rate=lr, 
            min_learning_rate=min_learning_rate, 
            warmup_iters=warmup_iters, 
            cosine_cycle_iters=cosine_cycle_iters
        )
        
        # 2. 将计算出的学习率注入优化器
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        (X,Y)=get_batch(data,batch_size,context_length,device)
        result=model.forward(X)
        loss=cross_entropy(rearrange(result,'batch seq vocab->(batch seq) vocab'),rearrange(Y,'batch seq->(batch seq)'))
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(),max_norm)
        optimizer.step()
        if save_every is not None:
            if i%save_every==0 and i!=0:
                save_checkpoint(model,optimizer,i,out)
        print(f'iter{i}:loss={loss.item()}')
    save_checkpoint(model,optimizer,iteration_step,out)
    return model

@torch.no_grad()
def decode(
    model, 
    prompt: torch.Tensor, 
    max_tokens: int, 
    temperature: float = 1.0, 
    top_p: float = 1.0, 
    eos_token_id: int = None
):
    """
    参数:
        model: Transformer 模型
        prompt: 输入提示词 Tensor, 形状为 (batch, seq_len)
        max_tokens: 最大生成长度
        temperature: 温度参数 tau (公式 23)
        top_p: 核采样阈值 p (公式 24)
        eos_token_id: 停止符 ID (如 <|endoftext|>)
    """
    model.eval()
    device = next(model.parameters()).device
    if isinstance(prompt, list):
        curr_input = torch.tensor(prompt, dtype=torch.long,device=device)
    else:
        curr_input = prompt.to(device)
    
    for _ in range(max_tokens):
        # logits 形状: (seq_len, vocab_size) -> 取 (-1, :)
        logits = model(curr_input)[-1, :]
        
        if temperature != 1.0:
            logits = logits / max(temperature, 1e-8)
        
        # 转化为概率分布
        probs = softmax(logits, i=-1)
        
        # 3. 应用 Top-p
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            current_sum = 0.0
            cutoff_idx = 0

            for i in range(len(sorted_probs)):
                current_sum += sorted_probs[i]
                cutoff_idx = i
                if current_sum >= top_p:
                    break
            sorted_probs[cutoff_idx + 1 :] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum()
            # 4. 从分布中采样下一个 Token
            next_token = sorted_indices[torch.multinomial(sorted_probs, num_samples=1)]
        else:
            next_token = torch.multinomial(probs, num_samples=1)
        
        # 5. 拼接到当前序列
        curr_input = torch.cat([curr_input, next_token], dim=0)
        
        # 6. 检查是否生成了 EOS 停止符
        if eos_token_id is not None and (next_token == eos_token_id).all():
            break
            
    return curr_input








if __name__ == "__main__":
    '''model=train(10000,256,512,4,16,1344,'/home/zhang/projects/cs336/cs336-assignment1/data/check.pt',128,10000,'/home/zhang/projects/cs336/cs336-assignment1/data/train_tokens.bin',1000,10000,device='cuda:0',dtype=torch.bfloat16)'''
    model=Transformer(10000,256,512,4,16,1344,10000,weights=None,device='cuda:0',dtype=torch.bfloat16)
    optimizer=AdamW(model.parameters())
    load_checkpoint('/home/zhang/projects/cs336/cs336-assignment1/model/check.pt',model,optimizer)
    tokenizer=Tokenizer.from_files('/home/zhang/projects/cs336/cs336-assignment1/bpe_tinystories/vocab_10000.json','/home/zhang/projects/cs336/cs336-assignment1/bpe_tinystories/merge_10000.txt',['<|endoftext|>'])
    prompt='''Deep in the heart of the Whispering Woods, there lived a tiny squirrel named Pip. Unlike the other squirrels who spent their days gathering brown nuts, Pip was a dreamer. One chilly autumn evening,'''
    encoded=tokenizer.encode(prompt)
    result=decode(model,encoded,150,1,0.9,0)
    print(tokenizer.decode(result))