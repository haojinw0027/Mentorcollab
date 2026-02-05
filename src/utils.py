"""
MLP model for branch prediction (Option A vs Option B).

This model takes hidden states from the language model and predicts
which branch (A or B) should be taken at the decision point.
"""

import torch
import torch.nn as nn
from torchvision.ops import MLP

MODEL_NAME_DICT = {
    "meta-llama/Llama-3.1-8B-Instruct": "llama-3.1-8b",
    "meta-llama/Llama-3.1-8B": "llama-3.1-8b-base",
    "meta-llama/Llama-3.1-70B-Instruct": "llama-3.1-70b",
    "meta-llama/Llama-3.1-70B": "llama-3.1-70b-base",
    "google/gemma-3-12b-pt": "gemma-3-12b-base",
    "google/gemma-3-12b-it": "gemma-3-12b",
    "mistralai/Mistral-Small-3.1-24B-Base-2503": "mistral-small-3.1-24b",
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503": "mistral-small-3.1-24b",
    "google/gemma-3-4b-pt": "gemma-3-4b-base",
    "google/gemma-3-4b-it": "gemma-3-4b",
    "google/gemma-3-1b-pt": "gemma-3-1b-base",
    "google/gemma-3-1b-it": "gemma-3-1b",
    "meta-llama/Llama-3.2-1B": 'llama-3.2-1B-base',
    "meta-llama/Llama-3.2-1B-Instruct": 'llama-3.2-1B',
    "meta-llama/Llama-3.2-3B": 'llama-3.2-3B-base',
    "meta-llama/Llama-3.2-3B-Instruct": 'llama-3.2-3B',
    "google/gemma-3-27b-it": "gemma-3-27b",
    "google/gemma-3-27b-pt": "gemma-3-27b-base",
    "Qwen/Qwen3-32B-FP8": "qwen-3.2-32b",
    "meta-llama/Llama-3.2-1B-Instruct": "llama-3.2-1b",
    "meta-llama/Llama-3.2-3B-Instruct": "llama-3.2-3b",
    "meta-llama/Llama-3.2-1B": "llama-3.2-1b-base",
    "meta-llama/Llama-3.2-3B": "llama-3.2-3b-base",
    "Qwen/Qwen3-14B-Base": "qwen3-14b-base",
    "Qwen/Qwen3-14B": "qwen3-14b",
    "Qwen/Qwen3-8B-Base": "qwen3-8b-base",
    "Qwen/Qwen3-8B": "qwen3-8b",
    "Qwen/Qwen3-4B-Base": "qwen3-4b-base",
    "Qwen/Qwen3-4B": "qwen3-4b",
    "Qwen/Qwen3-1.7B-Base": "qwen3-1.7b-base",
    "Qwen/Qwen3-1.7B": "qwen3-1.7b",
    "Qwen/Qwen3-0.6B-Base": "qwen3-0.6b-base",
    "Qwen/Qwen3-0.6B": "qwen3-0.6b",
    "Qwen/Qwen3-4B-Thinking-2507-FP8": "qwen3-4b-thinking-2507-fp8",
    "meta-llama/Llama-3.3-70B-Instruct": "llama-3.3-70b",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": "deepseek-r1-distill-llama-70b",
    "openai/gpt-oss-20b": "gpt-oss-20b",
    "nvidia/NVIDIA-Nemotron-Nano-12B-v2": "nvidia-nemotron-nano-12b-v2",
    "microsoft/Phi-4-reasoning": "phi-4-reasoning",
    "zai-org/GLM-4-32B-0414": "glm-4-32b-0414",
    "Qwen/Qwen3-32B": "qwen3-32b",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "deepseek-r1-distill-llama-8b",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B": "deepseek-r1-0528-qwen3-8b",
    "Qwen/QwQ-32B": "qwq-32b",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": "deepseek-r1-distill-qwen-14b",
    "Qwen/Qwen3-4B-Thinking-2507": "qwen3-4b-thinking-2507",
    "Qwen/Qwen2-72B": "qwen2-72b",
}

INSTRUCT_MODEL_LIST = [ "qwen3-32b", "qwq-32b", "glm-4-32b-0414", "qwen3-4b-thinking-2507", "deepseek-r1-distill-qwen-14b", "deepseek-r1-distill-llama-8b", "deepseek-r1-0528-qwen3-8b", "nvidia/NVIDIA-Nemotron-Nano-12B-v2", "microsoft/Phi-4-reasoning", "openai/gpt-oss-20b", "meta-llama/Llama-3.3-70B-Instruct", "Qwen/Qwen3-1.7B", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", "Qwen/Qwen3-0.6B", "Qwen/Qwen3-8B", "Qwen/Qwen3-4B", "Qwen/Qwen3-14B", "google/gemma-3-27b-it", "meta-llama/Llama-3.2-3B-Instruct", "meta-llama/Llama-3.2-1B-Instruct", "google/gemma-3-1b-it", "meta-llama/Llama-3.1-70B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", "google/gemma-3-12b-it", "mistralai/Mistral-Small-3.1-24B-Instruct-2503", "google/gemma-3-4b-it", "meta-llama/Llama-3.1-1B-Instruct", "meta-llama/Llama-3.1-3B-Instruct"]

def apply_instruct_template(model_name, system_prompt, instruct_prompt, response_prompt, add_bos=False):
    return f"{system_prompt}\n{instruct_prompt}\n{response_prompt}"


class BranchPredictionMLP(nn.Module):
    """
    MLP model for predicting which branch to take (A or B).

    Architecture:
        Input (hidden_size)
        → MLP (hidden_size → 2*hidden_size → hidden_size → hidden_size//2)
        → Linear (hidden_size//2 → 1)
        → Sigmoid

    Output:
        - Score between 0 and 1
        - > 0.5 means choose Option A
        - ≤ 0.5 means choose Option B
    """

    def __init__(self, hidden_size, dropout=0.1):
        """
        Args:
            hidden_size: Hidden size of the language model
            dropout: Dropout probability (default: 0.1)
        """
        super(BranchPredictionMLP, self).__init__()
        self.hidden_size = hidden_size

        # MLP layers with BatchNorm and ReLU
        self.mlp = MLP(
            in_channels=hidden_size,
            hidden_channels=[2 * hidden_size, hidden_size, hidden_size // 2],
            norm_layer=nn.BatchNorm1d,
            activation_layer=nn.ReLU,
            dropout=dropout,
        )

        # Final linear layer to produce single score
        self.linear = nn.Linear(hidden_size // 2, 1)

        # Sigmoid to get probability
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden_states):
        """
        Forward pass.

        Args:
            hidden_states: Tensor of shape [batch_size, hidden_size] or [batch_size, 1, hidden_size]

        Returns:
            Tensor of shape [batch_size] with scores between 0 and 1
        """
        # Handle both 2D and 3D inputs
        if hidden_states.dim() == 3:
            # Shape: [batch_size, seq_length, hidden_size]
            batch_size, seq_length, hidden_size = hidden_states.size()
            if seq_length != 1:
                raise ValueError(f"Expected seq_length=1, got {seq_length}")
            hidden_states = hidden_states.squeeze(1)  # [batch_size, hidden_size]
        elif hidden_states.dim() == 2:
            # Shape: [batch_size, hidden_size]
            pass
        else:
            raise ValueError(f"Expected 2D or 3D input, got shape {hidden_states.shape}")

        # Pass through MLP
        mlp_output = self.mlp(hidden_states)  # [batch_size, hidden_size // 2]

        # Linear layer
        logits = self.linear(mlp_output)  # [batch_size, 1]

        # Sigmoid activation
        scores = self.sigmoid(logits).squeeze(-1)  # [batch_size]

        return scores

    def predict(self, hidden_states, threshold=0.5):
        """
        Predict branch choice (A or B).

        Args:
            hidden_states: Tensor of shape [batch_size, hidden_size]
            threshold: Threshold for choosing A vs B (default: 0.5)

        Returns:
            List of 'A' or 'B' predictions
        """
        scores = self.forward(hidden_states)
        predictions = []

        for score in scores:
            if score.item() > threshold:
                predictions.append('A')
            else:
                predictions.append('B')

        return predictions


def create_model(model_config):
    """
    Create BranchPredictionMLP from a model config.

    Args:
        model_config: AutoConfig object from transformers or dict with 'hidden_size'

    Returns:
        BranchPredictionMLP instance
    """
    if hasattr(model_config, 'hidden_size'):
        hidden_size = model_config.hidden_size
    elif hasattr(model_config, 'text_config') and hasattr(model_config.text_config, 'hidden_size'):
        # Handle Gemma3 and other multimodal configs
        hidden_size = model_config.text_config.hidden_size
    elif isinstance(model_config, dict) and 'hidden_size' in model_config:
        hidden_size = model_config['hidden_size']
    else:
        raise ValueError("model_config must have 'hidden_size' attribute or key")

    return BranchPredictionMLP(hidden_size=hidden_size)


if __name__ == "__main__":
    # Test the model
    print("Testing BranchPredictionMLP...")

    # Test with Qwen3-8B hidden size (4096)
    hidden_size = 4096
    batch_size = 4

    model = BranchPredictionMLP(hidden_size=hidden_size)
    print(f"Model created with hidden_size={hidden_size}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass with 2D input
    hidden_states_2d = torch.randn(batch_size, hidden_size)
    scores = model(hidden_states_2d)
    print(f"\n2D input shape: {hidden_states_2d.shape}")
    print(f"Output scores shape: {scores.shape}")
    print(f"Output scores: {scores}")

    # Test forward pass with 3D input
    hidden_states_3d = torch.randn(batch_size, 1, hidden_size)
    scores = model(hidden_states_3d)
    print(f"\n3D input shape: {hidden_states_3d.shape}")
    print(f"Output scores shape: {scores.shape}")
    print(f"Output scores: {scores}")

    # Test predict method
    predictions = model.predict(hidden_states_2d)
    print(f"\nPredictions: {predictions}")

    print("\nModel test passed!")
