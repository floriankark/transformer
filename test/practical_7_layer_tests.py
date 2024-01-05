import os
import sys

import pytest
import torch

# Add the parent directory to the system path for importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modelling.functional import BaseTransformerLayer

# Define test data for hidden states and attention masks
INPUT = torch.tensor(
    [
        [
            [1.9269, 1.4873, 0.9007, -2.1055],
            [0.6784, -1.2345, -0.0431, -1.6047],
            [0.3559, -0.6866, -0.4934, 0.2415],
        ],
        [
            [-1.1109, 0.0915, -2.3169, -0.2168],
            [-0.3097, -0.3957, 0.8034, -0.6216],
            [0.0000, 0.0000, 0.0000, 0.0000],
        ],
    ]
)

ATTENTION_MASK = torch.tensor([[1, 1, 1], [1, 1, 0]])

# Define test data for attention outputs (feature_dim is the hidden dimension of the position wise feed forward layer)
TEST_DATA = [
    (
        BaseTransformerLayer(
            input_dim=INPUT.size(-1), num_heads=2, feature_dim=6, dropout=0.0
        ),
        INPUT,
        ATTENTION_MASK,
        torch.tensor(
            [
                [
                    [
                        0.7562403082847595,
                        -1.4328937530517578,
                        1.3348004817962646,
                        -0.6581472754478455,
                    ],
                    [
                        -0.3241388201713562,
                        -0.8887178897857666,
                        1.8732272386550903,
                        -0.6603702306747437,
                    ],
                    [
                        0.4691855013370514,
                        -1.8848323822021484,
                        0.8884169459342957,
                        0.5272297859191895,
                    ],
                ],
                [
                    [
                        1.4459928274154663,
                        -1.6236248016357422,
                        0.30913472175598145,
                        -0.1315028816461563,
                    ],
                    [
                        0.5192930102348328,
                        1.3882019519805908,
                        -1.5955982208251953,
                        -0.31189611554145813,
                    ],
                    [0.0, 0.0, 0.0, 0.0],
                ],
            ]
        ),
    )
]

# Define test data for multi-head attention state dictionary
STATE_DICT = {
    "self_attention.query_transform.weight": torch.tensor(
        [
            [1.0311, -0.7048, 1.0131, -0.3308],
            [0.5177, 0.3878, -0.5797, -0.1691],
            [-0.5733, 0.5069, -0.4752, -0.4920],
            [0.2704, -0.5628, 0.6793, 0.4405],
        ]
    ),
    "self_attention.key_transform.weight": torch.tensor(
        [
            [-0.3609, -0.0606, 0.0733, 0.8187],
            [1.4805, 0.3449, -1.4241, -0.1163],
            [0.2176, -0.0467, -1.4335, -0.5665],
            [-0.4253, 0.2625, -1.4391, 0.5214],
        ]
    ),
    "self_attention.value_transform.weight": torch.tensor(
        [
            [1.0414, -0.3997, -2.2933, 0.4976],
            [-0.4257, -1.3371, -0.1933, 0.6526],
            [-0.3063, -0.3302, -0.9808, 0.1947],
            [-1.6535, 0.6814, 1.4611, -0.3098],
        ]
    ),
    "self_attention.output_transform.weight": torch.tensor(
        [
            [0.9633, -0.3095, 0.5712, 1.1179],
            [-1.2956, 0.0503, -0.5855, -0.3900],
            [0.9812, -0.6401, -0.4908, 0.2080],
            [-1.1586, -0.9637, -0.3750, 0.8033],
        ]
    ),
    "feature_transformation.linear1.weight": torch.tensor(
        [
            [-0.6284, -0.6438, 1.5350, 1.1490],
            [0.2052, 0.2262, -1.7528, -0.3317],
            [-1.5973, -1.6713, -0.0838, 0.3023],
            [-1.8687, 1.3992, 0.4657, -0.4965],
            [0.3568, 0.7022, 1.6242, 0.0296],
            [-0.9825, -1.1156, -1.0796, -2.3653],
        ]
    ),
    "feature_transformation.linear1.bias": torch.tensor(
        [-0.6578, 0.0737, -1.1023, 1.0154, -0.0636, 0.6766]
    ),
    "feature_transformation.linear2.weight": torch.tensor(
        [
            [-0.1321, -0.8338, 1.9037, 1.3632, -0.5201, -0.1429],
            [0.2040, 1.9728, 1.1372, -0.3463, -0.2783, 0.0706],
            [0.7269, 0.6720, -0.7278, -0.0459, 0.9857, -0.2458],
            [0.2121, -0.2146, 1.7223, 0.5707, -0.2097, 0.5553],
        ]
    ),
    "feature_transformation.linear2.bias": torch.tensor(
        [0.2078, 0.2490, 0.0818, -0.0595]
    ),
    "layer_norm_1.weight": torch.tensor([1.0, 1.0, 1.0, 1.0]),
    "layer_norm_1.bias": torch.tensor([0.0, 0.0, 0.0, 0.0]),
    "layer_norm_2.weight": torch.tensor([1.1, 1.1, 1.1, 1.1]),
    "layer_norm_2.bias": torch.tensor([0.0, 0.0, 0.0, 0.0]),
}


# Multi-head Attention Layer Tests
@pytest.mark.parametrize(
    "layer, input, attention_mask, expected", TEST_DATA, ids=["encoder_layer"]
)
def test_layer(layer, input, attention_mask, expected):
    """Test the Multi-head Attention layer."""
    # Load pre-defined state dictionary into the multi-head attention layer
    layer.load_state_dict(STATE_DICT)

    actual = layer(input, attention_mask)
    # Mask padded positions
    actual *= attention_mask.unsqueeze(-1).float()

    assert torch.allclose(actual, expected)
