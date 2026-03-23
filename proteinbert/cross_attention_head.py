"""
Enhanced finetune head for ProteinBERT.

Replaces the original cross-attention mechanism (which was ineffective because
static global features cannot produce meaningful position-aware queries) with:

1. Self-attention pooling: learns which sequence positions matter for classification
2. Max pooling: captures the strongest signal at any position
3. Proper dimension reduction: projects high-dim hidden layers to manageable sizes
4. Gated fusion: combines sequence, global, and manual features
5. Residual classifier head: deep but regularized classification layers

Key insight from AcrPred paper: the model fusion + downsampling ensemble strategy
is at least as important as the architecture itself.
"""

import tensorflow as tf
from tensorflow import keras

from .conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs


def build_enhanced_model(pretrained_model_generator, seq_len, feature_dim,
                         dropout_rate = 0.5, proj_dim = 256):
    """Build enhanced finetune model with attention pooling + gated fusion.

    Args:
        pretrained_model_generator: ProteinBERT pretrained model generator
        seq_len: input sequence length
        feature_dim: dimension of manual features
        dropout_rate: dropout rate for regularization
        proj_dim: projection dimension for hidden representations

    Returns:
        (model, base_model) tuple
    """
    base_model = pretrained_model_generator.create_model(seq_len, compile = False, init_weights = True)
    base_model = get_model_with_hidden_layers_as_outputs(base_model)
    seq_output, global_output = base_model.output
    # seq_output: (batch, seq_len, D_seq ~1562)
    # global_output: (batch, D_global ~15599)

    manual_input = keras.layers.Input(shape = (feature_dim,), name = "manual-features")

    # ================================================================
    # 1. Self-attention pooling on sequence output
    #    Learns which positions are important for Acr classification
    # ================================================================
    attn_key = keras.layers.Dense(128, activation = "tanh", name = "attn-key")(seq_output)
    attn_score = keras.layers.Dense(1, name = "attn-score")(attn_key)
    attn_weight = keras.layers.Softmax(axis = 1, name = "attn-softmax")(attn_score)
    # Weighted sum over sequence positions
    weighted_seq = keras.layers.Multiply(name = "attn-weighted")([seq_output, attn_weight])
    pooled_seq = keras.layers.Lambda(
        lambda x: tf.reduce_sum(x, axis = 1), name = "attn-pool")(weighted_seq)

    # ================================================================
    # 2. Max pooling captures strongest signal at any position
    # ================================================================
    max_seq = keras.layers.GlobalMaxPooling1D(name = "max-pool")(seq_output)

    # ================================================================
    # 3. Project each representation source to manageable dimensions
    #    Critical: 15599-dim global output is way too large for ~1100 samples
    # ================================================================
    seq_repr = keras.layers.Concatenate(name = "seq-repr")([pooled_seq, max_seq])
    seq_proj = keras.layers.Dense(proj_dim, activation = "gelu", name = "seq-proj")(seq_repr)
    seq_proj = keras.layers.LayerNormalization(name = "seq-ln")(seq_proj)

    global_proj = keras.layers.Dense(proj_dim, activation = "gelu", name = "global-proj")(global_output)
    global_proj = keras.layers.LayerNormalization(name = "global-ln")(global_proj)

    manual_proj = keras.layers.Dense(proj_dim // 2, activation = "gelu", name = "manual-proj")(manual_input)
    manual_proj = keras.layers.LayerNormalization(name = "manual-ln")(manual_proj)

    # ================================================================
    # 4. Gated fusion of all information sources
    # ================================================================
    merged = keras.layers.Concatenate(name = "fusion")(
        [seq_proj, global_proj, manual_proj])
    merged = keras.layers.Dropout(dropout_rate, name = "fusion-drop")(merged)

    # ================================================================
    # 5. Residual classifier head
    # ================================================================
    # First residual block
    h = keras.layers.Dense(proj_dim, activation = "gelu", name = "res-dense1")(merged)
    h = keras.layers.Dropout(dropout_rate, name = "res-drop1")(h)
    h = keras.layers.Dense(proj_dim, name = "res-dense2")(h)
    shortcut = keras.layers.Dense(proj_dim, use_bias = False, name = "res-shortcut")(merged)
    h = keras.layers.Add(name = "res-add")([h, shortcut])
    h = keras.layers.LayerNormalization(name = "res-ln")(h)
    h = keras.layers.Activation("gelu", name = "res-act")(h)

    # Final classification layers
    h = keras.layers.Dense(64, activation = "gelu", name = "head-dense")(h)
    h = keras.layers.Dropout(dropout_rate * 0.5, name = "head-drop")(h)

    output = keras.layers.Dense(1, activation = "sigmoid", name = "output")(h)

    model = keras.models.Model(
        inputs = base_model.inputs + [manual_input], outputs = output)
    return model, base_model


# Keep backward-compatible alias
build_cross_attention_model = build_enhanced_model
