from keras.models import Model
from keras.layers import Input, Dense, LSTM, Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, concatenate, Dropout, Activation, Masking, Reshape, multiply
from utils.layer_utils import AttentionLSTM
from utils.constants import MAX_NB_VARIABLES, NB_CLASSES_LIST, MAX_TIMESTEPS_LIST

#48: Arousal, 49: Valence, 50: HighLow
DATASET_INDEX = 48
MAX_TIMESTEPS = MAX_TIMESTEPS_LIST[DATASET_INDEX]
MAX_NB_VARIABLES = MAX_NB_VARIABLES[DATASET_INDEX]
NB_CLASS = NB_CLASSES_LIST[DATASET_INDEX]

# Squeeze-Excite Block
def squeeze_excite_block(input):
    """
    Create a squeeze-excite block for recalibrating feature maps.
    
    Args:
    input (tensor): Input tensor to apply squeeze-excite mechanism on.

    Returns:
    tensor: Output tensor after applying the squeeze-excite block.
    """
    filters = input.shape[-1]  # Get the number of filters (channels)
    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input, se])
    return se


# Model 1: LSTM + Conv1D with Squeeze-Excite
def generate_model():
    """
    Generate a model combining LSTM and Conv1D layers for DEAP dataset classification.
    Uses Masking, LSTM, Conv1D, BatchNormalization, Squeeze-Excite, and GlobalAveragePooling1D layers.
  
    Returns:
    model: A compiled Keras model.
    """
    ip = Input(shape=(MAX_NB_VARIABLES, MAX_TIMESTEPS))
    
    # LSTM-based feature extraction
    x = Masking()(ip)
    x = LSTM(8)(x)  # LSTM layer with 8 units #change it 8, 64, 128
    x = Dropout(0.8)(x)

    # Conv1D-based feature extraction
    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = GlobalAveragePooling1D()(y)

    # Concatenation of LSTM and Conv1D features
    x = concatenate([x, y])

    # Final dense layer with softmax activation
    out = Dense(NB_CLASS, activation='softmax')(x)

    # Compile the model
    model = Model(ip, out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model


# Model 2: Attention LSTM + Conv1D with Squeeze-Excite
def generate_model_2():
    """
    Generate a model using Attention LSTM and Conv1D layers for DEAP dataset classification.
    This architecture uses Attention-based LSTM for temporal features and Conv1D for spatial features.
  
    Returns:
    model: A compiled Keras model.
    """

    ip = Input(shape=(MAX_NB_VARIABLES, MAX_TIMESTEPS))

    # Temporal feature extraction with Attention LSTM
    stride = 3  # Subsampling to prevent out-of-memory errors
    x = Permute((2, 1))(ip)
    x = Conv1D(MAX_NB_VARIABLES // stride, 8, strides=stride, padding='same', activation='relu', use_bias=False,
               kernel_initializer='he_uniform')(x)
    x = Masking()(x)
    x = AttentionLSTM(384, unroll=True)(x)
    x = Dropout(0.8)(x)

    # Spatial feature extraction with Conv1D
    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = squeeze_excite_block(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = GlobalAveragePooling1D()(y)

    # Concatenation of Attention LSTM and Conv1D features
    x = concatenate([x, y])

    # Final dense layer with softmax activation
    out = Dense(NB_CLASS, activation='softmax')(x)

    # Compile the model
    model = Model(ip, out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


# Model 3: LSTM + Conv1D without Squeeze-Excite
def generate_model_3():
    """
    Generate a model combining LSTM and Conv1D layers for DEAP dataset classification.
    Similar to Model 1 but without Squeeze-Excite blocks for comparison.
    
    Returns:
    model: A compiled Keras model.
    """

    ip = Input(shape=(MAX_NB_VARIABLES, MAX_TIMESTEPS))

    # LSTM-based feature extraction
    x = Masking()(ip)
    x = LSTM(8)(x)
    x = Dropout(0.8)(x)

    # Conv1D-based feature extraction
    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = GlobalAveragePooling1D()(y)

    # Concatenation of LSTM and Conv1D features
    x = concatenate([x, y])

    # Final dense layer with softmax activation
    out = Dense(NB_CLASS, activation='softmax')(x)

    # Compile the model
    model = Model(ip, out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


# Model 4: Attention LSTM + Conv1D without Squeeze-Excite
def generate_model_4():
    """
    Generate a model using Attention LSTM and Conv1D layers for DEAP dataset classification.
    Similar to Model 2 but without Squeeze-Excite blocks for comparison.
   
    Returns:
    model: A compiled Keras model.
    """

    ip = Input(shape=(MAX_NB_VARIABLES, MAX_TIMESTEPS))

    # Temporal feature extraction with Attention LSTM
    stride = 3  # Subsampling to prevent out-of-memory errors
    x = Permute((2, 1))(ip)
    x = Conv1D(MAX_NB_VARIABLES // stride, 8, strides=stride, padding='same', activation='relu', use_bias=False,
               kernel_initializer='he_uniform')(x)
    x = Masking()(x)
    x = AttentionLSTM(384, unroll=True)(x)
    x = Dropout(0.8)(x)

    # Spatial feature extraction with Conv1D
    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = GlobalAveragePooling1D()(y)

    # Concatenation of Attention LSTM and Conv1D features
    x = concatenate([x, y])

    # Final dense layer with softmax activation
    out = Dense(NB_CLASS, activation='softmax')(x)

    # Compile the model
    model = Model(ip, out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
