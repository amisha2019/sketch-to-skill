import math

def calculate_conv_output_shape(input_shape, conv_params):
    """
    Calculate the output shape of a convolutional layer.
    
    :param input_shape: The shape of the input (channels, height, width)
    :param conv_params: The parameters of the convolutional layer (out_channels, kernel_size, stride, padding)
    :return: The shape of the output (channels, height, width)
    """
    channels, height, width = input_shape
    out_channels, kernel_size, stride, padding = conv_params
    
    out_height = math.floor((height - kernel_size + 2 * padding) / stride + 1)
    out_width = math.floor((width - kernel_size + 2 * padding) / stride + 1)
    
    return (out_channels, out_height, out_width)

def calculate_transposed_conv_output_shape(input_shape, conv_params):
    """
    Calculate the output shape of a transposed convolutional layer.
    
    :param input_shape: The shape of the input (channels, height, width)
    :param conv_params: The parameters of the transposed convolutional layer (out_channels, kernel_size, stride, padding, output_padding)
    :return: The shape of the output (channels, height, width)
    """
    channels, height, width = input_shape
    out_channels, kernel_size, stride, padding, output_padding = conv_params
    
    out_height = (height - 1) * stride - 2 * padding + kernel_size + output_padding
    out_width = (width - 1) * stride - 2 * padding + kernel_size + output_padding
    
    return (out_channels, out_height, out_width)

def print_encoder_dimensions(input_shape, hidden_dims, img_size=64):
    """
    Print the output dimensions of each layer in the encoder.
    
    :param input_shape: The shape of the input image (channels, height, width)
    :param hidden_dims: List of channel numbers for the hidden layers
    """
    current_shape = input_shape
    print(f"Input shape: {current_shape}")
    
    for i, h_dim in enumerate(hidden_dims):
        if img_size == 64:
            conv_params = (h_dim, 3, 2, 1)  # (out_channels, kernel_size, stride, padding)
        elif img_size == 224:
            conv_params = (h_dim, 5, 3, 1)
        else:
            raise ValueError("Invalid image size")
        current_shape = calculate_conv_output_shape(current_shape, conv_params)
        print(f"After conv layer {i+1}: {current_shape}")
    
    # Calculate the flattened feature dimension
    flattened_dim = current_shape[0] * current_shape[1] * current_shape[2]
    print(f"Flattened dimension: {flattened_dim}\n")

def print_decoder_dimensions(latent_shape, hidden_dims, img_size=64):
    """
    Print the output dimensions of each layer in the decoder.
    
    :param latent_dim: The dimension of the latent space
    :param hidden_dims: List of channel numbers for the hidden layers
    """
    current_shape = latent_shape
    print(f"Latent shape: {current_shape}")
    pdd = [2, 0, 2, 2]
    
    for i, h_dim in enumerate(hidden_dims):
        if img_size == 64:
            conv_params = (h_dim, 3, 2, 1, 1)  # (out_channels, kernel_size, stride, padding, output_padding)
        elif img_size == 224:
            conv_params = (h_dim, 5, 3, 1, pdd[i])
        else:
            raise ValueError("Invalid image size")
        current_shape = calculate_transposed_conv_output_shape(current_shape, conv_params)
        print(f"After deconv layer {i+1}: {current_shape}; output padding: {conv_params[-1]}")

    # Final deconv layers
    if img_size == 64:
        conv_params = (hidden_dims[-1], 3, 2, 1, 1)
    elif img_size == 224:
        conv_params = (hidden_dims[-1], 5, 3, 1, 2)
    current_shape = calculate_transposed_conv_output_shape(current_shape, conv_params)
    print(f"After deconv layer {len(hidden_dims)+1}: {current_shape}; output padding: {conv_params[-1]}")
    
    conv_params = (1, 3, 1, 1, 0)
    current_shape = calculate_transposed_conv_output_shape(current_shape, conv_params)
    print(f"After final deconv layer: {current_shape}\n")


def calculate_vae_model_size_64():
    # For image size 64x64
    # Use the function to calculate the dimensions of your encoder
    input_shape = (1, 64, 64)
    hidden_dims = [32, 64, 128, 256, 512]
    print_encoder_dimensions(input_shape, hidden_dims, img_size=64)

    # Use the function to calculate the dimensions of your decoder
    latent_shape = (512, 2, 2)
    hidden_dims = [256, 128, 64, 32]
    print_decoder_dimensions(latent_shape, hidden_dims, img_size=64)

    # Input shape: (1, 64, 64)
    # After conv layer 1: (32, 32, 32)
    # After conv layer 2: (64, 16, 16)
    # After conv layer 3: (128, 8, 8)
    # After conv layer 4: (256, 4, 4)
    # After conv layer 5: (512, 2, 2)
    # Flattened dimension: 2048

    # Latent shape: (512, 2, 2)
    # After deconv layer 1: (256, 4, 4); output padding: 1
    # After deconv layer 2: (128, 8, 8); output padding: 1
    # After deconv layer 3: (64, 16, 16); output padding: 1
    # After deconv layer 4: (32, 32, 32); output padding: 1
    # After deconv layer 5: (32, 64, 64); output padding: 1
    # After final deconv layer: (1, 64, 64)

def calculate_vae_model_size_224():
    # For image size 224x224
    # Use the function to calculate the dimensions of your encoder
    input_shape = (1, 224, 224)
    hidden_dims = [16, 32, 64, 128]
    print_encoder_dimensions(input_shape, hidden_dims, img_size=224)

    # Use the function to calculate the dimensions of your decoder
    latent_dim = (128, 2, 2)
    hidden_dims = [64, 32, 16]
    print_decoder_dimensions(latent_dim, hidden_dims, img_size=224)

    # Input shape: (1, 224, 224)
    # After conv layer 1: (16, 74, 74)
    # After conv layer 2: (32, 24, 24)
    # After conv layer 3: (64, 8, 8)
    # After conv layer 4: (128, 2, 2)
    # Flattened dimension: 512

    # Latent shape: (128, 2, 2)
    # After deconv layer 1: (64, 8, 8); output padding: 2
    # After deconv layer 2: (32, 24, 24); output padding: 0
    # After deconv layer 3: (16, 74, 74); output padding: 2
    # After deconv layer 4: (16, 224, 224); output padding: 2
    # After final deconv layer: (1, 224, 224)


if __name__ == "__main__":
    calculate_vae_model_size_64()
    # calculate_vae_model_size_224()
    