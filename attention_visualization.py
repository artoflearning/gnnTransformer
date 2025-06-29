import torch  # Import PyTorch for deep learning operations
import matplotlib.pyplot as plt  # Import plotting library
import seaborn as sns  # Import statistical data visualization
import numpy as np  # Import numerical computing library
from custom_attention import CustomAttention  # Import our custom attention implementation

def visualize_attention_weights(attention_weights, title="Attention Weights", save_path=None):
    """
    Visualize attention weights using a heatmap.
    
    Args:
        attention_weights (torch.Tensor): Attention weights tensor of shape [seq_len, seq_len]
        title (str): Title for the plot
        save_path (str, optional): Path to save the visualization
    """
    # Convert PyTorch tensor to numpy array and move to CPU if needed
    weights = attention_weights.detach().cpu().numpy()
    # Handle batch dimension if present
    if len(weights.shape) > 2:
        weights = weights[0]  # Take first batch if multiple batches
    
    # Create figure with specified size
    plt.figure(figsize=(10, 8))
    # Create heatmap with annotations showing exact values
    sns.heatmap(weights, cmap='viridis', annot=True, fmt='.2f')
    plt.title(title)  # Set plot title
    plt.xlabel('Key Position')  # Label x-axis
    plt.ylabel('Query Position')  # Label y-axis
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path)
    plt.show()  # Display the plot

def visualize_attention_patterns(attention_weights, input_tokens=None, output_tokens=None, save_path=None):
    """
    Visualize attention patterns with token labels.
    
    Args:
        attention_weights (torch.Tensor): Attention weights tensor
        input_tokens (list, optional): List of input tokens
        output_tokens (list, optional): List of output tokens
        save_path (str, optional): Path to save the visualization
    """
    # Convert to numpy and handle batch dimension
    weights = attention_weights.detach().cpu().numpy()
    if len(weights.shape) > 2:
        weights = weights[0]
    
    # Create figure for visualization
    plt.figure(figsize=(12, 10))
    # Create heatmap without annotations for cleaner look
    sns.heatmap(weights, cmap='viridis')
    
    # Add token labels if provided
    if input_tokens and output_tokens:
        plt.xticks(np.arange(len(input_tokens)) + 0.5, input_tokens, rotation=45)
        plt.yticks(np.arange(len(output_tokens)) + 0.5, output_tokens, rotation=0)
    
    plt.title("Attention Pattern Visualization")
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_attention_heads(attention_weights, num_heads, save_path=None):
    """
    Visualize attention patterns across multiple heads.
    
    Args:
        attention_weights (torch.Tensor): Attention weights tensor of shape [num_heads, seq_len, seq_len]
        num_heads (int): Number of attention heads
        save_path (str, optional): Path to save the visualization
    """
    # Convert to numpy array
    weights = attention_weights.detach().cpu().numpy()
    
    # Create subplot for each attention head
    fig, axes = plt.subplots(1, num_heads, figsize=(20, 4))
    for i in range(num_heads):
        sns.heatmap(weights[i], ax=axes[i], cmap='viridis')
        axes[i].set_title(f'Head {i+1}')
    
    plt.tight_layout()  # Adjust layout to prevent overlap
    if save_path:
        plt.savefig(save_path)
    plt.show()

# Example usage
if __name__ == "__main__":
    # Create sample attention weights for demonstration
    seq_len = 5  # Length of sequence
    attention_weights = torch.randn(seq_len, seq_len)  # Random weights
    attention_weights = torch.softmax(attention_weights, dim=-1)  # Convert to probabilities
    
    # Visualize single attention pattern
    visualize_attention_weights(attention_weights, "Sample Attention Weights")
    
    # Visualize with token labels
    sample_tokens = ["The", "cat", "sat", "on", "mat"]
    visualize_attention_patterns(attention_weights, sample_tokens, sample_tokens)
    
    # Visualize multiple attention heads
    num_heads = 4  # Number of attention heads
    multi_head_weights = torch.randn(num_heads, seq_len, seq_len)  # Random weights for multiple heads
    multi_head_weights = torch.softmax(multi_head_weights, dim=-1)  # Convert to probabilities
    plot_attention_heads(multi_head_weights, num_heads)

    # Example with custom attention mechanism
    attention = CustomAttention(d_model=64)  # Create attention mechanism
    query = torch.randn(1, 5, 64)  # Random query tensor [batch_size, seq_len, d_model]
    key = torch.randn(1, 5, 64)    # Random key tensor
    value = torch.randn(1, 5, 64)  # Random value tensor

    # Compute attention and get weights
    output = attention(query, key, value)

    # Visualize the attention pattern from first batch
    visualize_attention_weights(attention.attention_weights[0], "My Attention Pattern")