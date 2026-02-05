from tensorflow import keras

from .conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs


def build_cross_attention_model(pretrained_model_generator, seq_len, feature_dim, n_heads = 4, key_dim = 64, dropout_rate = 0.5):
    base_model = pretrained_model_generator.create_model(seq_len, compile = False, init_weights = True)
    base_model = get_model_with_hidden_layers_as_outputs(base_model)
    seq_output, global_output = base_model.output

    manual_input = keras.layers.Input(shape = (feature_dim,), name = "manual-features")
    q = keras.layers.Dense(n_heads * key_dim, activation = "gelu")(manual_input)
    q = keras.layers.Reshape((1, n_heads * key_dim))(q)
    attn = keras.layers.MultiHeadAttention(num_heads = n_heads, key_dim = key_dim, name = "cross-attention")(q, seq_output, seq_output)
    attn = keras.layers.Flatten()(attn)

    merged = keras.layers.Concatenate()([attn, global_output, manual_input])
    merged = keras.layers.Dropout(dropout_rate)(merged)
    output = keras.layers.Dense(1, activation = "sigmoid")(merged)

    model = keras.models.Model(inputs = base_model.inputs + [manual_input], outputs = output)
    return model, base_model
