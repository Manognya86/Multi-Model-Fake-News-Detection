import matplotlib.pyplot as plt
import numpy as np

def visualize_attention(text, attention_weights, tokenizer):
    """
    Visualize attention weights for a given text
    """
    tokens = tokenizer.tokenize(text)
    attention_weights = attention_weights[:len(tokens), :len(tokens)]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(attention_weights, cmap='viridis')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_yticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    ax.set_yticklabels(tokens)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Attention Weight', rotation=-90, va='bottom')
    
    plt.title('Attention Weights Visualization')
    plt.tight_layout()
    plt.show()