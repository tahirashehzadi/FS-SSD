import torch
import torch.nn as nn

def enhanced_representation(x1, x2):
    x1 = torch.randn(4, 30, 256)
    x2 = torch.randn(4, 300, 256)
    x1 = x1.to(device)
    x2 = x2.to(device)
    batch_size, l1, s = x1.size(0), x1.size(1), x1.size(2)
    
    linear_layer = nn.Linear(256, 64).to(device)
    x1 = x1.view(-1, s)  # Reshape to (4*30, 256)
    x1 = linear_layer(x1)  # Apply the linear layer
    x1 = x1.view(4, l1, 64)
    
    batch_size, l2, s = x2.size(0), x2.size(1), x2.size(2)
    x2 = x2.view(-1, s)  # Reshape to (4*30, 256)
    x2 = linear_layer(x2)  # Apply the linear layer
    x2 = x2.view(4, l2, 64)
    

    orig_x1 = x1
    batch_size, l1, s = x1.size(0), x1.size(1), x1.size(2)
    l2 = x2.size(1)

    # Define the Linear Layer
    m1 = nn.Linear(l2 * s, l1 * s)

    # Reshape the tensor to shape (4, b*256)
    x2_flattened = x2.view(batch_size, -1)

    # Apply the linear transformation
    x2_transformed = m1(x2_flattened)

    # Reshape the tensor back to shape (4, 30, 256)
    x2_final = x2_transformed.view(batch_size, l1, s)

    x = torch.cat((x1, x2_final), dim=1)

    # Calculating the number of attention blocks
    batch_size, img_n, c_embed = x.size()
    num_temporal_attention_blocks = 8

    # Using x for embedding
    x_embed = x

    # Performing multi-head attention
    x_embed = x_embed.view(batch_size, img_n, num_temporal_attention_blocks, -1)
    target_x_embed = x_embed[:, [1]]
    ada_weights = torch.sum(x_embed * target_x_embed, dim=3, keepdim=True) / (float(c_embed / num_temporal_attention_blocks)**0.5)
    ada_weights = ada_weights.expand(-1, -1, -1, int(c_embed / num_temporal_attention_blocks)).contiguous()
    ada_weights = ada_weights.view(batch_size, img_n, c_embed)
    ada_weights = ada_weights.softmax(dim=1)

    # Aggregation and generation of enhanced representation
    x = (x * ada_weights).sum(dim=1)

    x = x.view(batch_size, 1, s, 1)  # Size becomes [batch_size, 1, s, 1]
    # Define the upsampling layer
    upsample = nn.UpsamplingBilinear2d((orig_x1.size()[-2], orig_x1.size()[-1]))
    # Upsample x
    x_upsampled = upsample(x)
    # Remove the channel dimension
    x_upsampled = x_upsampled.squeeze(1)
    aggregated_enhanced_representation = orig_x1 + x_upsampled
    # Calculate the difference in the second dimension
    diff_dim = x2.size(1) - aggregated_enhanced_representation.size(1)

    # Check if there's a difference to handle
    if diff_dim > 0:
          #zeros_to_add = torch.zeros((aggregated_enhanced_representation.size(0), diff_dim, aggregated_enhanced_representation.size(2))).to(device)
          zeros_to_add = torch.zeros((aggregated_enhanced_representation.size(0), diff_dim, aggregated_enhanced_representation.size(2))).to(device)
          # Concatenate the zero tensor with aggregated_enhanced_representation
          aggregated_enhanced_representation = torch.cat((aggregated_enhanced_representation, zeros_to_add), dim=1)
    
    bx1=aggregated_enhanced_representation
    batch_size, bl1, s = bx1.size(0), bx1.size(1), bx1.size(2)
    
    
    linear_layer_back = nn.Linear(64, 256).to(device)
    bx1 = bx1.view(-1, s)  # Reshape to (4*30, 256)
    bx1 = linear_layer_back(bx1)  # Apply the linear layer
    bx1 = bx1.view(4, bl1, 256)

    print("Size of bx1:", bx1.shape)
    return bx1

# Sample tensors
x1 = torch.randn(4, 30, 256)
x2 = torch.randn(4, 300, 256)

# Call the function
result = enhanced_representation(x1, x2)
print(f"aggregated_enhanced_representation: {result.size()}")
