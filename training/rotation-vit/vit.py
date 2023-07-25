import torch
import torch.nn as nn
import numpy as np


def patchify(images: torch.Tensor, n_patches: int) -> torch.Tensor:
    n, c, h, w = images.shape
    assert h % n_patches == 0, "H must be divisible by n_patches"
    assert w % n_patches == 0, "W must be divisible by n_patches"
    patches = torch.zeros(n, n_patches**2, h * w * c // n_patches**2).to(
        images.device
    )
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[
                    :,
                    i * patch_size : (i + 1) * patch_size,
                    j * patch_size : (j + 1) * patch_size,
                ]
                patches[idx, i * n_patches + j] = patch.flatten()
                # patches[idx, i*n_patches+j] = images[idx, :, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size].flatten()
    return patches


def get_positional_embeddings(sequence_length: int, d: int) -> torch.Tensor:
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = (
                np.sin(i / (10000 ** (j / d)))
                if j % 2 == 0
                else np.cos(i / (10000 ** ((j - 1) / d)))
            )
    return result


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention"""

    def __init__(self, d, n_heads=2) -> None:
        super(MultiHeadAttention, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert (
            self.d % self.n_heads == 0
        ), f"Can't divide dimension {d} into {n_heads} heads"

        d_head = self.d // self.n_heads
        print(f"Using {n_heads} heads of dimension {d_head}")

        self.q_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)]
        )
        self.k_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)]
        )
        self.v_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)]
        )
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

        print("Initialized MSA")
        print(f"Q: {self.q_mappings}")
        print(f"K: {self.k_mappings}")
        print(f"V: {self.v_mappings}")
        print(f"Softmax: {self.softmax}")

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        # Sequence has shape (N, seq_len, token_dim)
        # We go into shape   ( N, seq_len, n_heads, token_dim // n_heads)
        # And come back to   (N, seq_len, item_dim) (through concatenation)
        results = []
        for sequence in sequences:
            seq_results = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]
                seq = sequence[:, head * self.d_head : (head + 1) * self.d_head]

                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)
                attention = self.softmax(q @ k.T / (self.d_head**0.5))
                seq_results.append(attention @ v)

            results.append(torch.hstack(seq_results))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in results])


class MyVitBlock(nn.Module):
    """ViT Block : Residual + Multi-Head Self-Attention + MLP"""

    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyVitBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(self.hidden_d)
        self.mhsa = MultiHeadAttention(self.hidden_d, self.n_heads)
        self.norm2 = nn.LayerNorm(self.hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, mlp_ratio * self.hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * self.hidden_d, self.hidden_d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out


class ViTForImageRotation(nn.Module):
    def __init__(
        self, chw=(1, 28, 28), n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10
    ) -> None:
        super(ViTForImageRotation, self).__init__()

        self.chw = chw  # C, H, W

        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d

        # Input shape must be divisible by n_patches

        assert self.chw[1] % self.n_patches == 0, "H must be divisible by n_patches"
        assert self.chw[2] % self.n_patches == 0, "W must be divisible by n_patches"
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # 1. Add linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # 2. Leranable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # 3. Add positional embeddings(+1 for class token)
        # self.pos_embed = nn.Parameter(get_positional_embeddings(n_patches**2 + 1, self.hidden_d))
        # self.pos_embed.requires_grad = False
        self.register_buffer(
            "positional_embeddings",
            get_positional_embeddings(n_patches**2 + 1, self.hidden_d),
            persistent=False,
        )

        # 4. Add transformer blocks
        self.block = nn.ModuleList(
            [MyVitBlock(self.hidden_d, self.n_heads) for _ in range(self.n_blocks)]
        )

        # 5. Add MLP head
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_features=out_d), nn.Softmax(dim=-1)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images has shape (N, C, H, W)
        n, c, h, w = images.shape
        patches = patchify(images, self.n_patches)

        # Running through linear mapper
        # Map the vector corresponding to each patch to the hidden size dimension
        tokens = self.linear_mapper(patches)

        # Adding classification token to the tokens
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        # tokens = torch.stack([torch.vstack((self.class_token, token)) for token in tokens])
        # tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])

        # add positional embeddings
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)

        # Run through transformer blocks
        for block in self.block:
            out = block(out)

        # Run through MLP head
        cls_token = out[:, 0, :]  # Get the classification token
        out = self.mlp(cls_token)

        return out
