import numpy as np


def conv2d(input, kernel, padding):
    """
    Multi-channel convolution operation
    between feature map and conv kernel
    :param input: input feature as N * C * H * W np.array
    :param kernel: conv kernel as np.array
    """
    batch_size = input.shape[0]
    input_dim, output_dim = kernel.shape[1], kernel.shape[0]
    input_h, input_w = input.shape[2], input.shape[3]
    kernel_h, kernel_w = kernel.shape[2], kernel.shape[3]
    output_h = input_h + 2 * padding - kernel_h + 1
    output_w = input_w + 2 * padding - kernel_w + 1

    # Padding input feature map
    padding_h = input_h + 2 * padding
    padding_w = input_w + 2 * padding
    input_padding = np.zeros((batch_size, input_dim, padding_h, padding_w))
    input_padding[..., padding:input_h+padding, padding:input_w+padding] = input

    # Expand convolution kernel from C'*C*K*K to N*C'*C*K*K
    kernel = kernel[np.newaxis, ...].repeat(batch_size, axis=0)
    # Expand padded feature map from N*C*H*W to N*C'*C*H*W
    input_padding = input_padding[:, np.newaxis, ...].repeat(output_dim, axis=1)

    output = np.zeros((batch_size, output_dim, output_h, output_w))
    for i in range(0, padding_h-kernel_h + 1):
        for j in range(0, padding_w-kernel_w + 1):
            output[..., i, j] = \
                np.sum(input_padding[..., i:i+kernel_h, j:j+kernel_w]*kernel, axis=(2, 3, 4))

    return output


def conv2d_deriv(input1, input2, padding):
    """
    Multi-channel convolution operation
    between feature map and derivation map
    :param input1: input1 feature as N * C * H * W np.array
    :param input2: derivation as N * C' * H' * W' np.array
    """
    batch_size = input1.shape[0]
    # print(input1.shape, input2.shape)
    input_dim, output_dim = input1.shape[1], input2.shape[1]
    input1_h, input1_w = input1.shape[2], input1.shape[3]
    input2_h, input2_w = input2.shape[2], input2.shape[3]
    output_h = input1_h + 2 * padding - input2_h + 1
    output_w = input1_w + 2 * padding - input2_w + 1

    # Padding input1 feature map
    padding_h = input1_h + 2 * padding
    padding_w = input1_w + 2 * padding
    input1_padding = np.zeros((batch_size, input_dim, padding_h, padding_w))
    input1_padding[..., padding:input1_h+padding, padding:input1_w+padding] = input1

    # Expand input2 from N*C'*H'*W' to N*C'*C*H'*W'
    input2 = input2[:, :, np.newaxis, ...].repeat(input_dim, axis=2)
    # Expand padded input1 from N*C*H*W to N*C'*C*H*W
    input1_padding = input1_padding[:, np.newaxis, ...].repeat(output_dim, axis=1)

    output = np.zeros((batch_size, output_dim, input_dim, output_h, output_w))
    for i in range(0, padding_h-input2_h + 1):
        for j in range(0, padding_w-input2_w + 1):
            output[..., i, j] = \
                np.sum(input1_padding[..., i:i+input2_h, j:j+input2_w]*input2, axis=(3, 4))

    return output


def maxpool(input, kernel_size, padding, stride):
    """
    Max pooling operation
    :param input: input feature as N * C * H * W np.array
    :param kernel_size: pooling kernel size
    :param padding: zero padding
    :param padding: stride of pooling
    """
    batch_size, input_dim = input.shape[0], input.shape[1]
    input_h, input_w = input.shape[2], input.shape[3]
    output_h = int((input_h + 2 * padding - kernel_size) / stride) + 1
    output_w = int((input_w + 2 * padding - kernel_size) / stride) + 1

    # Padding input feature map
    padding_h = input_h + 2 * padding
    padding_w = input_w + 2 * padding
    input_padding = np.zeros((batch_size, input_dim, padding_h, padding_w))
    input_padding[..., padding:input_h+padding, padding:input_w+padding] = input

    output = np.zeros((batch_size, input_dim, output_h, output_w))
    pos_record = np.zeros((batch_size, input_dim, output_h, output_w, 4), np.int)
    for i in range(0, padding_h-kernel_size + 1, stride):
        for j in range(0, padding_w-kernel_size + 1, stride):
            output_i, output_j = int(i / stride), int(j / stride)
            grid = input_padding[..., i:i + kernel_size, j:j + kernel_size]
            max_val = np.max(grid, axis=(2, 3))
            output[..., output_i, output_j] = max_val

            for batch_idx in range(batch_size):
                for dim_idx in range(input_dim):
                    dh, dw = np.unravel_index(grid[batch_idx, dim_idx].argmax(), grid[batch_idx, dim_idx].shape)
                    pos_record[batch_idx, dim_idx, output_i, output_j] = [batch_idx, dim_idx, i+dh, j+dw]

            # print(max_val)
            # max_val = max_val[..., np.newaxis, np.newaxis].repeat(padding_h, axis=-2).repeat(padding_w, axis=-1)
            # print(max_val.shape)
            # print(np.argwhere(max_val == input_padding).shape)
            # max_pos = np.argwhere(max_val == input_padding).reshape(batch_size, input_dim, -1)

    pos_record = pos_record - padding

    return output, pos_record

