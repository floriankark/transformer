import os
import sys

import pytest
import torch

# Add the parent directory to the system path for importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modelling.attention import MultiHeadAttention

# Define test data for hidden states and attention masks
VALUE = torch.tensor(
    [
        [[0.0349, 0.3211, 1.5736, -0.8455], [0.0000, 0.0000, 0.0000, 0.0000]],
        [[-1.4181, 0.8963, 0.0499, 2.2667], [1.1790, -0.4345, -1.3864, -1.2862]],
    ]
)

QUERY = torch.tensor(
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

QUERY_ATTENTION_MASK = torch.tensor([[1, 1, 1], [1, 1, 0]])
VALUE_ATTENTION_MASK = torch.tensor([[1, 0], [1, 1]])

# Define test data for attention outputs
MHA_TEST_DATA = [
    (
        MultiHeadAttention(QUERY.size(-1), 2, mask_future=False),
        QUERY,
        QUERY,
        QUERY_ATTENTION_MASK,
        torch.tensor(
            [
                [
                    [
                        -0.2657615542411804,
                        -1.1095069646835327,
                        1.575529932975769,
                        -2.953308582305908,
                    ],
                    [
                        -0.31810081005096436,
                        -1.3583225011825562,
                        1.0969884395599365,
                        -3.5557737350463867,
                    ],
                    [
                        -1.0363497734069824,
                        0.1544792652130127,
                        0.8307660818099976,
                        -0.6420033574104309,
                    ],
                ],
                [
                    [
                        3.41835618019104,
                        -3.4588632583618164,
                        2.471224546432495,
                        -2.2303943634033203,
                    ],
                    [
                        -0.835513710975647,
                        0.5645291805267334,
                        -2.178706645965576,
                        -0.2814210057258606,
                    ],
                    [
                        1.3362171649932861,
                        -1.6601970195770264,
                        0.15386708080768585,
                        -1.618969202041626,
                    ],
                ],
            ]
        ),
    ),
    (
        MultiHeadAttention(QUERY.size(-1), 2, mask_future=True),
        QUERY,
        QUERY,
        QUERY_ATTENTION_MASK,
        torch.tensor(
            [
                [
                    [
                        -1.8750097751617432,
                        3.4548227787017822,
                        2.242863178253174,
                        6.896119117736816,
                    ],
                    [
                        -1.439692735671997,
                        1.0199611186981201,
                        0.8919554948806763,
                        0.4862484931945801,
                    ],
                    [
                        -1.0363497734069824,
                        0.1544792652130127,
                        0.8307660818099976,
                        -0.6420033574104309,
                    ],
                ],
                [
                    [
                        3.5263192653656006,
                        -6.098883628845215,
                        1.9741934537887573,
                        -7.373747825622559,
                    ],
                    [
                        -0.835513710975647,
                        0.5645291805267334,
                        -2.178706645965576,
                        -0.2814210057258606,
                    ],
                    [
                        1.3362171649932861,
                        -1.6601970195770264,
                        0.15386708080768585,
                        -1.618969202041626,
                    ],
                ],
            ]
        ),
    ),
    (
        MultiHeadAttention(QUERY.size(-1), 2, mask_future=False),
        QUERY,
        VALUE,
        VALUE_ATTENTION_MASK,
        torch.tensor(
            [
                [
                    [
                        -1.5669221878051758,
                        5.281073570251465,
                        -1.7499525547027588,
                        8.89909839630127,
                    ],
                    [
                        -1.5669221878051758,
                        5.281073570251465,
                        -1.7499525547027588,
                        8.89909839630127,
                    ],
                    [
                        -1.5669221878051758,
                        5.281073570251465,
                        -1.7499525547027588,
                        8.89909839630127,
                    ],
                ],
                [
                    [
                        0.13075685501098633,
                        -4.142998218536377,
                        2.939213275909424,
                        -7.535215854644775,
                    ],
                    [
                        -0.38490769267082214,
                        0.6016770601272583,
                        -1.6096503734588623,
                        0.05100858211517334,
                    ],
                    [
                        0.9850694537162781,
                        -2.1259379386901855,
                        0.8979899883270264,
                        -2.8792083263397217,
                    ],
                ],
            ]
        ),
    ),
]

# Define test data for multi-head attention state dictionary
STATE_DICT = {
    "query_transform.weight": torch.tensor(
        [
            [1.0311, -0.7048, 1.0131, -0.3308],
            [0.5177, 0.3878, -0.5797, -0.1691],
            [-0.5733, 0.5069, -0.4752, -0.4920],
            [0.2704, -0.5628, 0.6793, 0.4405],
        ]
    ),
    "key_transform.weight": torch.tensor(
        [
            [-0.3609, -0.0606, 0.0733, 0.8187],
            [1.4805, 0.3449, -1.4241, -0.1163],
            [0.2176, -0.0467, -1.4335, -0.5665],
            [-0.4253, 0.2625, -1.4391, 0.5214],
        ]
    ),
    "value_transform.weight": torch.tensor(
        [
            [1.0414, -0.3997, -2.2933, 0.4976],
            [-0.4257, -1.3371, -0.1933, 0.6526],
            [-0.3063, -0.3302, -0.9808, 0.1947],
            [-1.6535, 0.6814, 1.4611, -0.3098],
        ]
    ),
    "output_transform.weight": torch.tensor(
        [
            [0.9633, -0.3095, 0.5712, 1.1179],
            [-1.2956, 0.0503, -0.5855, -0.3900],
            [0.9812, -0.6401, -0.4908, 0.2080],
            [-1.1586, -0.9637, -0.3750, 0.8033],
        ]
    ),
}


# Multi-head Attention Layer Tests
@pytest.mark.parametrize(
    "mha_layer, query, value, attention_mask, expected",
    MHA_TEST_DATA,
    ids=["self-attention", "self-attention-future-masked", "cross-attention"],
)
def test_multi_head_attention(mha_layer, query, value, attention_mask, expected):
    """Test the Multi-head Attention layer."""
    # Load pre-defined state dictionary into the multi-head attention layer
    mha_layer.load_state_dict(STATE_DICT)

    assert torch.allclose(mha_layer(query, value, value, attention_mask), expected)
