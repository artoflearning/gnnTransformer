import torch  # Import PyTorch for deep learning operations
import torch.nn as nn  # Import neural network modules
import torch.nn.functional as F  # Import functional interface for neural network operations
import math  # Import math for mathematical operations

class CustomAttention(nn.Module):
    """
    A custom implementation of the attention mechanism from scratch.
    This implementation follows the scaled dot-product attention mechanism
    as described in "Attention Is All You Need" (Vaswani et al., 2017).
    """
    
    def __init__(self, d_model, dropout=0.1):
        """
        Initialize the custom attention mechanism.
        
        Args:
            d_model (int): The dimension of the model (size of input/output vectors)
            dropout (float): Probability of dropping neurons during training (default: 0.1)
        """
        super().__init__()  # Initialize the parent class (nn.Module)
        self.d_model = d_model  # Store the model dimension
        self.dropout = nn.Dropout(dropout)  # Create dropout layer for regularization
        
        # Create linear layers for projecting inputs into Query, Key, and Value spaces
        self.q_proj = nn.Linear(d_model, d_model)  # Query projection
        self.k_proj = nn.Linear(d_model, d_model)  # Key projection
        self.v_proj = nn.Linear(d_model, d_model)  # Value projection
        
        # Create output projection layer
        self.out_proj = nn.Linear(d_model, d_model)  # Output projection
        
    def forward(self, query, key, value, mask=None):
        """
        Compute attention scores and weighted sum of values.
        
        Args:
            query (torch.Tensor): Query tensor of shape [batch_size, seq_len, d_model]
            key (torch.Tensor): Key tensor of shape [batch_size, seq_len, d_model]
            value (torch.Tensor): Value tensor of shape [batch_size, seq_len, d_model]
            mask (torch.Tensor, optional): Mask tensor to prevent attention to certain positions
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, d_model]
        """
        batch_size = query.size(0)  # Get batch size from input
        
        # Project inputs into Query, Key, and Value spaces
        q = self.q_proj(query)  # Transform query input
        k = self.k_proj(key)    # Transform key input
        v = self.v_proj(value)  # Transform value input
        
        # Compute attention scores using scaled dot-product
        # Shape: [batch_size, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)
        
        # Apply mask if provided (set masked positions to very negative value)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Convert scores to probabilities using softmax
        self.attention_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout to attention weights for regularization
        attention_weights = self.dropout(self.attention_weights)
        
        # Compute weighted sum of values using attention weights
        # Shape: [batch_size, seq_len, d_model]
        output = torch.matmul(attention_weights, v)
        
        # Project output to final dimension
        output = self.out_proj(output)
        
        return output  # Return the final output

# Example usage
if __name__ == "__main__":
    # Create sample input tensors
    batch_size = 2  # Number of samples in batch
    seq_len = 5     # Length of sequence
    d_model = 64    # Dimension of model
    
    # Create random input tensors
    query = torch.randn(batch_size, seq_len, d_model)  # Random query tensor
    key = torch.randn(batch_size, seq_len, d_model)    # Random key tensor
    value = torch.randn(batch_size, seq_len, d_model)  # Random value tensor
    
    # Create attention mechanism
    attention = CustomAttention(d_model)
    
    # Compute attention
    output = attention(query, key, value)
    
    # Print shapes for verification
    print(f"Input shape: {query.shape}")
    print(f"Output shape: {output.shape}") 