import torch
import torch.nn as nn
from braindecode.models import EEGConformer as BraindecoderEEGConformer


class EEGConformer(nn.Module):
    """
    EEG-Conformer model wrapper for BCI motor imagery classification.
    Wraps braindecode's EEGConformer with standard training interface.
    
    Parameters:
    -----------
    n_chans : int
        Number of EEG channels (default: 22 for BCI Competition IV 2a)
    n_outputs : int
        Number of motor imagery classes (default: 4)
    n_times : int
        Number of time points per sample (default: 500 for 2sec @ 250Hz)
    drop_prob : float
        Dropout probability (default: 0.5)
    num_layers : int
        Number of transformer layers (default: 6)
    num_heads : int
        Number of attention heads (default: 10)
    """
    
    def __init__(self, n_chans=22, n_outputs=4, n_times=500, drop_prob=0.5, 
                 num_layers=6, num_heads=10):
        super(EEGConformer, self).__init__()
        
        self.model = BraindecoderEEGConformer(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
            drop_prob=drop_prob,
            num_layers=num_layers,
            num_heads=num_heads
        )
    
    def forward(self, x):
        """
        Forward pass through EEG-Conformer.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch, channels, time_points) or (batch, 1, channels, time_points)
        
        Returns:
        --------
        logits : torch.Tensor
            Classification logits of shape (batch, n_outputs)
        """
        # Handle 3D input: (batch, channels, time) -> add channel dimension
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # (batch, 1, channels, time)
        
        return self.model(x)
