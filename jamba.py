from torch import Tensor, nn
from torch.nn import MultiheadAttention
from mamba_ssm import Mamba


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, activation: str = "relu"):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransformerMoEBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        num_experts: int,
        num_experts_per_token: int,
        *args,
        **kwargs,
    ):
        """
        Initializes a TransformerMoEBlock.

        Args:
            dim (int): The dimension of the input tensor.
            heads (int): The number of attention heads.
            num_experts (int): The total number of experts.
            num_experts_per_token (int): The number of experts per token.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_token

        self.attn = MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.moe = nn.ModuleList([
            FeedForward(dim, dim * 4, activation="relu") for _ in range(num_experts)
        ])

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the TransformerMoEBlock.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying the TransformerMoEBlock.
        """
        skip = x
        x = nn.LayerNorm(self.dim).to(x.device)(x)
        attn_output, _ = self.attn(x, x, x)
        x = attn_output + skip

        x = nn.LayerNorm(self.dim).to(x.device)(x)
        moe_out = sum(m(x) for m in self.moe[:self.num_experts_per_tok]) / self.num_experts_per_tok
        x = moe_out + skip
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        *args,
        **kwargs,
    ):
        """
        Initializes a TransformerBlock.

        Args:
            dim (int): Dimension of the input tensor.
            heads (int): Number of attention heads.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()
        self.dim = dim
        self.heads = heads

        self.attn = MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.ffn = FeedForward(dim, dim * 2, activation="relu")

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the TransformerBlock.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying the TransformerBlock.
        """
        skip = x
        x = nn.LayerNorm(self.dim).to(x.device)(x)
        attn_output, _ = self.attn(x, x, x)
        x = attn_output + skip

        skip_two = x
        x = nn.LayerNorm(self.dim).to(x.device)(x)
        x = self.ffn(x) + skip_two
        return x


class MambaMoELayer(nn.Module):
    def __init__(
        self,
        dim: int,
        d_state: int,
        d_conv: int,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        *args,
        **kwargs,
    ):
        """
        Initialize the MambaMoELayer.

        Args:
            dim (int): Dimension of the input tensor.
            d_state (int): Dimension of the state tensor.
            d_conv (int): Dimension of the convolutional tensor.
            num_experts (int, optional): Number of experts. Defaults to 8.
            num_experts_per_token (int, optional): Number of experts per token. Defaults to 2.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_token

        # Mamba
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=2)

        # MoE
        self.moe = nn.ModuleList([
            FeedForward(dim, dim * 2, activation="relu") for _ in range(num_experts)
        ])

    def forward(self, x: Tensor):
        """
        Forward pass of the MambaMoELayer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying the MambaMoELayer.
        """
        skip = x

        x = nn.LayerNorm(self.dim).to(x.device)(x)
        x = self.mamba(x) + x

        x = nn.LayerNorm(self.dim).to(x.device)(x)
        moe_out = sum(m(x) for m in self.moe[:self.num_experts_per_tok]) / self.num_experts_per_tok
        x = moe_out + skip
        return x


class JambaBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        d_state: int,
        d_conv: int,
        heads: int,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.heads = heads
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_token

        self.mamba_layer = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=2)

        self.mamba_moe_layer = MambaMoELayer(
            dim,
            d_state,
            d_conv,
            num_experts,
            num_experts_per_token,
        )

        self.transformer = TransformerBlock(
            dim,
            heads,
        )

    def forward(self, x: Tensor) -> Tensor:
        # x = self.mamba_layer(x)
        x = self.mamba_moe_layer(x)
        # x = self.transformer(x)
        # x = self.mamba_moe_layer(x)
        # x = self.mamba_layer(x)
        # x = self.mamba_moe_layer(x)
        return x


class Jamba(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_tokens: int,
        d_state: int,
        d_conv: int,
        heads: int,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        pre_emb_norm: bool = False,
        return_embeddings: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.d_state = d_state
        self.d_conv = d_conv
        self.heads = heads
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_token
        self.pre_emb_norm = pre_emb_norm
        self.return_embeddings = return_embeddings

        self.layers = nn.ModuleList(
            [
                JambaBlock(
                    dim,
                    d_state,
                    d_conv,
                    heads,
                    num_experts,
                    num_experts_per_token,
                )
                for _ in range(depth)
            ]
        )

        # self.embed = nn.Embedding(num_tokens, dim)
        self.norm = nn.LayerNorm(dim) if pre_emb_norm else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        # x = x.long()
        # x = self.embed(x)
        # print(x.shape)
        x = self.norm(x)
        for layer in self.layers:
            x = layer(x)
        return x


# if __name__ == '__main__':
#     model = Jamba(
#         dim=512,
#         depth=4,
#         num_tokens=101,
#         d_state=32,
#         d_conv=4,
#         heads=1,
#         num_experts=8,
#         num_experts_per_token=2,
#         pre_emb_norm=True,
#         return_embeddings=False,
#     )
#     import torch
#     # Take shape (B,L,D) AND i want output shape (B,L,D)
#     # Ignore all warnings
#     device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)

#     x = torch.randn(2, 10, 512).to(device)
#     # print(x.shape)
#     y = model(x)  # shape (B,L,D)
#     # print(y.shape)