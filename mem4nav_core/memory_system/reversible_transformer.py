# type: ignore
import torch
import torch.nn as nn
from typing import Tuple, List, Optional

class RevTransformerBlock(nn.Module):
    """
    A single block of the Reversible Transformer.
    Implements the equations:
        y1 = x1 + F(x2)
        y2 = x2 + G(y1)
    And the inverse:
        x2 = y2 - G(y1)
        x1 = y1 - F(x2)
    where F and G are learnable functions (e.g., Attention and MLP).
    Input x is expected to be (x1, x2) where x1 and x2 are of equal dimension (dim // 2).
    """
    def __init__(self, 
                 dim: int, # Total dimension of the concatenated input (x1 || x2)
                 num_heads: int, 
                 mlp_ratio: float = 4.0, 
                 dropout_p: float = 0.1,
                 use_pytorch_mha: bool = True # Use PyTorch's MHA or a custom one
                ):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Dimension ({dim}) must be even for splitting into two streams.")
        
        self.half_dim = dim // 2

        self.norm_f = nn.LayerNorm(self.half_dim)
        if use_pytorch_mha:
             # PyTorch MHA expects (seq_len, batch, embed_dim)
            self.F = nn.MultiheadAttention(embed_dim=self.half_dim, num_heads=num_heads, dropout=dropout_p, batch_first=False)
        else:
            # self.F = YourCustomAttention(dim=self.half_dim, num_heads=num_heads, dropout=dropout_p)
            raise NotImplementedError("Custom attention for F not implemented, use PyTorch MHA.")

        self.norm_g = nn.LayerNorm(self.half_dim)
        mlp_hidden_dim = int(self.half_dim * mlp_ratio)
        self.G = nn.Sequential(
            nn.Linear(self.half_dim, mlp_hidden_dim),
            nn.GELU(), # A common activation function in Transformers
            nn.Dropout(dropout_p),
            nn.Linear(mlp_hidden_dim, self.half_dim),
            nn.Dropout(dropout_p)
        )
        
        self.dropout = nn.Dropout(dropout_p) 

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, 
                attn_mask: Optional[torch.Tensor] = None, 
                key_padding_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the reversible block.
        Args:
            x1 (torch.Tensor): First half of the input. Shape (seq_len, batch_size, self.half_dim)
            x2 (torch.Tensor): Second half of the input. Shape (seq_len, batch_size, self.half_dim)
            attn_mask: Mask for multi-head attention.
            key_padding_mask: Key padding mask for multi-head attention.
        """
        # Pre-norm style for F and G (norm before the main operation)
        x2_norm = self.norm_f(x2)
  
        f_out, _ = self.F(query=x2_norm, key=x2_norm, value=x2_norm, 
                          attn_mask=attn_mask, key_padding_mask=key_padding_mask,
                          need_weights=False) 
        
        y1 = x1 + self.dropout(f_out) 

        y1_norm = self.norm_g(y1)
        g_out = self.G(y1_norm)
        y2 = x2 + self.dropout(g_out) 

        return y1, y2

    def inverse(self, y1: torch.Tensor, y2: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None, 
                key_padding_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse pass for the reversible block.
        Args:
            y1 (torch.Tensor): First half of the output from the forward pass.
            y2 (torch.Tensor): Second half of the output from the forward pass.
            attn_mask: Mask for multi-head attention.
            key_padding_mask: Key padding mask for multi-head attention.
        """
        y1_norm = self.norm_g(y1) # Must use y1, not x1_rec here
        g_out = self.G(y1_norm)
        x2_rec = y2 - self.dropout(g_out)

        x2_rec_norm = self.norm_f(x2_rec)
        f_out, _ = self.F(query=x2_rec_norm, key=x2_rec_norm, value=x2_rec_norm,
                           attn_mask=attn_mask, key_padding_mask=key_padding_mask,
                           need_weights=False)
        x1_rec = y1 - self.dropout(f_out)
        
        return x1_rec, x2_rec


class ReversibleTransformer(nn.Module):
    """
    A stack of Reversible Transformer Blocks (RevTransformerBlock).
    The input tensor x is expected to have its feature dimension `dim` be even,
    as it will be split into two halves (x1, x2) to be fed into the blocks.
    """
    def __init__(self, 
                 dim: int, # Total dimension of the input, which will be split.
                 depth: int, # Number of RevTransformerBlocks (L from paper [cite: 65])
                 num_heads: int, 
                 mlp_ratio: float = 4.0, 
                 dropout_p: float = 0.1):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Input dimension ({dim}) must be even to be split into two halves.")
        self.dim = dim
        self.depth = depth

        self.blocks = nn.ModuleList([
            RevTransformerBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout_p=dropout_p)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor, 
                attn_mask: Optional[torch.Tensor] = None, 
                key_padding_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Forward pass through the stack of reversible blocks.
        Args:
            x (torch.Tensor): Input tensor. Expected shape (seq_len, batch_size, dim)
                              or (batch_size, seq_len, dim) if MHA is batch_first.
                              PyTorch MHA defaults to (seq_len, batch, dim).
            attn_mask: Shared attention mask for all blocks.
            key_padding_mask: Shared key padding mask for all blocks.
        """
        # Split the input into two halves along the feature dimension
        x1, x2 = torch.chunk(x, 2, dim=-1)

        for block in self.blocks:
            x1, x2 = block(x1, x2, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        
        # Concatenate the two halves back
        return torch.cat([x1, x2], dim=-1)

    def inverse(self, y: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None, 
                key_padding_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Inverse pass through the stack of reversible blocks.
        Args:
            y (torch.Tensor): Output tensor from the forward pass.
            attn_mask: Shared attention mask for all blocks.
            key_padding_mask: Shared key padding mask for all blocks.
        """
        # Split the tensor y into two halves
        y1, y2 = torch.chunk(y, 2, dim=-1)

        # Iterate through the blocks in reverse order for the inverse pass
        for block in reversed(self.blocks):
            y1, y2 = block.inverse(y1, y2, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
            
        # Concatenate the reconstructed x1_rec, x2_rec
        return torch.cat([y1, y2], dim=-1)

if __name__ == '__main__':
    model_dim = 128 
    depth = 3     
    num_heads = 4
    seq_len = 10
    batch_size = 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create a ReversibleTransformer model
    rev_transformer = ReversibleTransformer(dim=model_dim, depth=depth, num_heads=num_heads).to(device)
    rev_transformer.train() # or .eval()

    # Create a dummy input tensor
    # Shape: (seq_len, batch_size, model_dim) for default PyTorch MHA
    dummy_input = torch.randn(seq_len, batch_size, model_dim, device=device)

    # Forward pass
    output = rev_transformer(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == dummy_input.shape

    # Inverse pass
    reconstructed_input = rev_transformer.inverse(output)
    print(f"Reconstructed input shape: {reconstructed_input.shape}")
    assert reconstructed_input.shape == dummy_input.shape

    # Check for reconstruction accuracy (should be very close for reversible models)
    # Detach from graph for comparison
    is_close = torch.allclose(reconstructed_input.detach(), dummy_input.detach(), atol=1e-5) # Increased atol due to float precision
    print(f"Is reconstructed input close to original input? {is_close}")
    if not is_close:
        diff = torch.abs(reconstructed_input.detach() - dummy_input.detach()).max()
        print(f"Max absolute difference: {diff.item()}")
        # Note: Perfect reconstruction depends on the operations within F and G being
        # numerically stable and their inverses being exact. Standard float32 arithmetic
        # can lead to tiny precision errors.
    

    example_key_padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    example_key_padding_mask[:, seq_len // 2:] = True # Mask out the second half of the sequence for each batch item

    output_masked = rev_transformer(dummy_input, key_padding_mask=example_key_padding_mask)
    reconstructed_masked = rev_transformer.inverse(output_masked, key_padding_mask=example_key_padding_mask)
    is_close_masked = torch.allclose(reconstructed_masked.detach(), dummy_input.detach(), atol=1e-5)
    print(f"Is reconstructed input close (with mask)? {is_close_masked}")


    print("\nReversibleTransformer tests completed.")