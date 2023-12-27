import sys
import os

import pytest
import torch

# Add the parent directory to the system path for importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Specify the pat to import your attention mechanism
from modelling.attention import Attention

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
ATTENTION_TEST_DATA = [
    (
        Attention(mask_future=False),
        QUERY,
        QUERY,
        QUERY_ATTENTION_MASK,
        torch.tensor(
            [
                [
                    [
                        1.9050642251968384,
                        1.4421212673187256,
                        0.8837929368019104,
                        -2.0934135913848877,
                    ],
                    [
                        0.9809781312942505,
                        -0.45746949315071106,
                        0.16625413298606873,
                        -1.5649594068527222,
                    ],
                    [
                        0.7208702564239502,
                        -0.5859572887420654,
                        -0.1027706041932106,
                        -0.8587384223937988,
                    ],
                ],
                [
                    [
                        -1.0970195531845093,
                        0.08305942267179489,
                        -2.2628417015075684,
                        -0.2238130271434784,
                    ],
                    [
                        -0.47443872690200806,
                        -0.2955244183540344,
                        0.16181963682174683,
                        -0.5383670330047607,
                    ],
                    [
                        -0.7103000283241272,
                        -0.15209999680519104,
                        -0.7567499876022339,
                        -0.41920000314712524,
                    ],
                ],
            ]
        ),
    ),
    (
        Attention(mask_future=True),
        QUERY,
        QUERY,
        QUERY_ATTENTION_MASK,
        torch.tensor(
            [
                [
                    [
                        1.926900029182434,
                        1.4873000383377075,
                        0.9006999731063843,
                        -2.1054999828338623,
                    ],
                    [
                        1.0457112789154053,
                        -0.4337407648563385,
                        0.23456794023513794,
                        -1.7520363330841064,
                    ],
                    [
                        0.7208702564239502,
                        -0.5859572887420654,
                        -0.1027706041932106,
                        -0.8587384223937988,
                    ],
                ],
                [
                    [
                        -1.1109000444412231,
                        0.09149999916553497,
                        -2.3169000148773193,
                        -0.2168000042438507,
                    ],
                    [
                        -0.47443872690200806,
                        -0.2955244183540344,
                        0.16181963682174683,
                        -0.5383670330047607,
                    ],
                    [
                        -0.7103000283241272,
                        -0.15209999680519104,
                        -0.7567499876022339,
                        -0.41920000314712524,
                    ],
                ],
            ]
        ),
    ),
    (
        Attention(mask_future=False),
        QUERY,
        VALUE,
        VALUE_ATTENTION_MASK,
        torch.tensor(
            [
                [
                    [
                        0.0348999984562397,
                        0.32109999656677246,
                        1.5736000537872314,
                        -0.8454999923706055,
                    ],
                    [
                        0.0348999984562397,
                        0.32109999656677246,
                        1.5736000537872314,
                        -0.8454999923706055,
                    ],
                    [
                        0.0348999984562397,
                        0.32109999656677246,
                        1.5736000537872314,
                        -0.8454999923706055,
                    ],
                ],
                [
                    [
                        0.22614431381225586,
                        0.0537601113319397,
                        -0.8594327569007874,
                        0.017331182956695557,
                    ],
                    [
                        0.12951618432998657,
                        0.10327407717704773,
                        -0.8059935569763184,
                        0.14952099323272705,
                    ],
                    [
                        -0.11954998970031738,
                        0.23090000450611115,
                        -0.6682499647140503,
                        0.49024999141693115,
                    ],
                ],
            ]
        ),
    ),
]


# Attention Layer Tests
@pytest.mark.parametrize(
    "attention_layer, query, value, attention_mask, expected",
    ATTENTION_TEST_DATA,
    ids=["self-attention", "self-attention-future-masked", "cross-attention"],
)
def test_attention(attention_layer, query, value, attention_mask, expected):
    """Test the Attention layer."""
    assert torch.allclose(
        attention_layer(query, value, value, attention_mask), expected
    )
