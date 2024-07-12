
`models.py`  

## DiT 类
class DiT(nn.Module):


### PatchEmbed
1、对图像patch划分。
```
self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
```
### TimestepEmbedder类
对时间步进行embedding，输入时间t，包含N步，输出(N,D)。输出维度：hidden_size

```
self.t_embedder = TimestepEmbedder(hidden_size)
t = self.t_embedder(t)  
```

### LabelEmbedder
类别标签嵌入为向量表示。
1、使用方法
```
self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
y = self.y_embedder(y, self.training)
```

2、实现
包含dropout则多一行类别
```
self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)

```
### DiTBlock

1、使用
```
self.blocks = nn.ModuleList([
    DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
])
for block in self.blocks:
    x = block(x, c) # c:condition embedding
```
2、实现
```
class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1) # condition 用于layer norm参数以及残差连接系数gate_msa以及gate_mlp
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
```