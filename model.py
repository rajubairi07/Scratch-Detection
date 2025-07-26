import tensorflow as tf
from tensorflow.keras import layers, Model

def conv_block(input_tensor, num_filters):
    """A block of two 3x3 convolution layers with ReLU activation and Dropout."""
    x = layers.Conv2D(num_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(input_tensor)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv2D(num_filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    return x

def encoder_block(input_tensor, num_filters):
    """An encoder block consists of a conv_block followed by max pooling."""
    conv = conv_block(input_tensor, num_filters)
    pool = layers.MaxPooling2D((2, 2))(conv)
    return conv, pool

def decoder_block(input_tensor, skip_features, num_filters):
    """A decoder block consists of up-sampling, concatenation with skip connection, and a conv_block."""
    up_conv = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
    concat = layers.concatenate([up_conv, skip_features])
    conv = conv_block(concat, num_filters)
    return conv

def build_unet_model(input_shape=(128, 128, 1)):
    """
    Builds and compiles the complete U-Net model.
    """
    inputs = layers.Input(shape=input_shape)

    # --- Encoder Path ---
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    # --- Bottleneck ---
    bottleneck = conv_block(p4, 1024)

    # --- Decoder Path ---
    d1 = decoder_block(bottleneck, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    # --- Output Layer ---
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(d4)
    model = Model(inputs=inputs, outputs=outputs, name="U-Net")
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
