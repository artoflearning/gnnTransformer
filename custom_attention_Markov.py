import torch  # Import PyTorch for deep learning operations
import torch.nn as nn  # Import neural network modules
import torch.nn.functional as F  # Import functional interface for neural network operations
import math  # Import math for mathematical operations

class CustomAttentionMarkov(nn.Module):
    """
    A custom implementation of the attention mechanism with Markov Chain Order 1 properties.
    This implementation follows the scaled dot-product attention mechanism with causal masking
    and log probability computation for numerical stability.
    """
    
    def __init__(self, d_model, dropout=0.1, use_log_probs=True):
        """
        Initialize the custom attention mechanism with Markov properties.
        
        Args:
            d_model (int): The dimension of the model (size of input/output vectors)
            dropout (float): Probability of dropping neurons during training (default: 0.1)
            use_log_probs (bool): Whether to use log probabilities for numerical stability
        """
        super().__init__()  # Initialize the parent class (nn.Module)
        self.d_model = d_model  # Store the model dimension
        self.dropout = nn.Dropout(dropout)  # Create dropout layer for regularization
        self.use_log_probs = use_log_probs  # Whether to use log probabilities
        
        # Create linear layers for projecting inputs into Query, Key, and Value spaces
        self.q_proj = nn.Linear(d_model, d_model)  # Query projection
        self.k_proj = nn.Linear(d_model, d_model)  # Key projection
        self.v_proj = nn.Linear(d_model, d_model)  # Value projection
        
        # Create output projection layer
        self.out_proj = nn.Linear(d_model, d_model)  # Output projection

    def create_causal_mask(self, seq_len):
        """
        Create a causal mask for Markov Chain Order 1 behavior.
        This ensures each position can only attend to previous positions.
        
        Args:
            seq_len (int): Length of the sequence
            
        Returns:
            torch.Tensor: Causal mask of shape [seq_len, seq_len]
        """
        # Create a matrix of ones
        mask = torch.ones(seq_len, seq_len)
        # Set upper triangular part to 0 (including diagonal)
        mask = torch.triu(mask, diagonal=1).bool()
        # Invert the mask (1 where attention is allowed, 0 where it's not)
        mask = ~mask
        return mask
        
    def forward(self, query, key, value, mask=None, use_causal_mask=True):
        """
        Compute attention scores and weighted sum of values with Markov properties.
        
        Args:
            query (torch.Tensor): Query tensor of shape [batch_size, seq_len, d_model]
            key (torch.Tensor): Key tensor of shape [batch_size, seq_len, d_model]
            value (torch.Tensor): Value tensor of shape [batch_size, seq_len, d_model]
            mask (torch.Tensor, optional): Additional mask tensor
            use_causal_mask (bool): Whether to use causal masking for Markov Chain Order 1
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, d_model]
        """
        batch_size = query.size(0)  # Get batch size from input
        seq_len = query.size(1)     # Get sequence length
        
        # Project inputs into Query, Key, and Value spaces
        q = self.q_proj(query)  # Transform query input
        k = self.k_proj(key)    # Transform key input
        v = self.v_proj(value)  # Transform value input
        
        # Compute attention scores using scaled dot-product
        # Shape: [batch_size, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_model)
        
        # Apply causal mask for Markov Chain Order 1
        if use_causal_mask:
            causal_mask = self.create_causal_mask(seq_len).to(scores.device)
            scores = scores.masked_fill(~causal_mask, -1e9)
        
        # Apply additional mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Convert scores to probabilities using log_softmax for numerical stability
        if self.use_log_probs:
            # Use log_softmax for numerical stability
            log_attention_weights = F.log_softmax(scores, dim=-1)
            self.attention_weights = torch.exp(log_attention_weights)
        else:
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
    
    # Create attention mechanism with log probabilities
    attention = CustomAttentionMarkov(d_model, use_log_probs=True)
    
    # Compute attention with causal masking (Markov Chain Order 1)
    output = attention(query, key, value)
    
    # Print shapes for verification
    print(f"Input shape: {query.shape}")
    print(f"Output shape: {output.shape}")
    
    # Print attention weights to verify causal masking
    print("\nAttention weights (first batch):")
    print(attention.attention_weights[0])
    
    # Verify Markov property (each position only attends to previous positions)
    print("\nVerifying Markov property (should be lower triangular):")
    print(attention.attention_weights[0] > 0)  # Show which positions can attend to each other 