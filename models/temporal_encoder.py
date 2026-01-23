"""
Temporal Encoder using Bidirectional ConvLSTM
Captures temporal dependencies in video sequences
"""
import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM Cell
    """
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super(ConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        
        # Convolutional gates
        self.conv = nn.Conv1d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )
        
    def forward(self, x, hidden_state):
        """
        Args:
            x: Input tensor (batch_size, input_dim, 1)
            hidden_state: Tuple of (h, c)
        Returns:
            h_next, c_next: Next hidden and cell states
        """
        h, c = hidden_state
        
        # Concatenate input and hidden state
        combined = torch.cat([x, h], dim=1)  # (B, input_dim + hidden_dim, 1)
        
        # Convolutional gates
        gates = self.conv(combined)  # (B, 4 * hidden_dim, 1)
        
        # Split into gates
        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)
        
        # Apply activations
        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        o = torch.sigmoid(o)  # Output gate
        g = torch.tanh(g)     # Cell gate
        
        # Update cell state
        c_next = f * c + i * g
        
        # Update hidden state
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden and cell states"""
        h = torch.zeros(batch_size, self.hidden_dim, 1, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, 1, device=device)
        return (h, c)


class TemporalEncoder(nn.Module):
    """
    Bidirectional ConvLSTM for temporal encoding
    """
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2):
        super(TemporalEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Forward LSTM layers
        self.forward_cells = nn.ModuleList([
            ConvLSTMCell(
                input_dim if i == 0 else hidden_dim,
                hidden_dim
            ) for i in range(num_layers)
        ])
        
        # Backward LSTM layers
        self.backward_cells = nn.ModuleList([
            ConvLSTMCell(
                input_dim if i == 0 else hidden_dim,
                hidden_dim
            ) for i in range(num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output projection (bidirectional)
        self.output_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            output: Encoded tensor of shape (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Forward pass
        forward_outputs = []
        for layer_idx in range(self.num_layers):
            hidden_state = self.forward_cells[layer_idx].init_hidden(batch_size, device)
            layer_output = []
            
            for t in range(seq_len):
                x_t = x[:, t, :].unsqueeze(-1) if layer_idx == 0 else forward_outputs[layer_idx-1][t]
                h, c = self.forward_cells[layer_idx](x_t, hidden_state)
                hidden_state = (h, c)
                layer_output.append(h)
            
            forward_outputs.append(layer_output)
        
        # Backward pass
        backward_outputs = []
        for layer_idx in range(self.num_layers):
            hidden_state = self.backward_cells[layer_idx].init_hidden(batch_size, device)
            layer_output = []
            
            for t in reversed(range(seq_len)):
                x_t = x[:, t, :].unsqueeze(-1) if layer_idx == 0 else backward_outputs[layer_idx-1][seq_len-1-t]
                h, c = self.backward_cells[layer_idx](x_t, hidden_state)
                hidden_state = (h, c)
                layer_output.append(h)
            
            layer_output.reverse()
            backward_outputs.append(layer_output)
        
        # Combine forward and backward outputs
        outputs = []
        for t in range(seq_len):
            forward_h = forward_outputs[-1][t].squeeze(-1)  # (B, hidden_dim)
            backward_h = backward_outputs[-1][t].squeeze(-1)  # (B, hidden_dim)
            combined = torch.cat([forward_h, backward_h], dim=1)  # (B, 2*hidden_dim)
            output = self.output_proj(combined)  # (B, hidden_dim)
            outputs.append(output)
        
        # Stack outputs
        outputs = torch.stack(outputs, dim=1)  # (B, seq_len, hidden_dim)
        
        # Apply dropout
        outputs = self.dropout(outputs)
        
        return outputs


if __name__ == "__main__":
    # Test the temporal encoder
    model = TemporalEncoder(input_dim=1280, hidden_dim=512, num_layers=2)
    x = torch.randn(4, 16, 1280)  # (batch, seq_len, features)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
