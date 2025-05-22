import torch
import torch.nn as nn
import gc
import asyncio

async def enhanced_representation(input_query_label_1, input_query_label_2):
        #await asyncio.sleep(10)  # Sleep for 1 second 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x1 = input_query_label_1.to(device)
        x2 = input_query_label_2.to(device)
        x1 = x1.to(device)
        x2 = x2.to(device)
        print("Size of input_query_label_1:", x1.shape)
        print("Size of input_query_label_2:", x2.shape)
        batch_size, l1, s = x1.size(0), x1.size(1), x1.size(2)
    
        linear_layer = nn.Linear(s, 64).to(device)
        x1 = x1.view(-1, s)  # Reshape to (4*30, 256)
        x1 = linear_layer(x1)  # Apply the linear layer
        x1 = x1.view(batch_size, l1, 64)
    
        batch_size, l2, s = x2.size(0), x2.size(1), x2.size(2)
        x2 = x2.view(-1, s)  # Reshape to (4*30, 256)
        x2 = linear_layer(x2)  # Apply the linear layer
        x2 = x2.view(batch_size, l2, 64)
        del linear_layer

        orig_x1 = x1
        batch_size, l1, s = x1.size(0), x1.size(1), x1.size(2)
        l2 = x2.size(1)

        # Define the Linear Layer
        m1 = nn.Linear(l2 * s, l1 * s).to(device)

        # Reshape the tensor to shape (4, b*256)
        x2_flattened = x2.view(batch_size, -1)

        # Apply the linear transformation
        x2_transformed = m1(x2_flattened)
        del m1
        # Reshape the tensor back to shape (4, 30, 256)
        x2_final = x2_transformed.view(batch_size, l1, s)
         
        x = torch.cat((x1, x2_final), dim=1).to(device)

        # Calculating the number of attention blocks
        batch_size, img_n, c_embed = x.size()
        num_temporal_attention_blocks = 8

        # Using x for embedding
        x_embed = x
        #print(x_embed.size())
        print("Size of x_embed:", x_embed.shape)
        # Performing multi-head attention
        x_embed = x_embed.view(batch_size, img_n, num_temporal_attention_blocks, -1)
        target_x_embed = x_embed[:, [1]]
        ada_weights = torch.sum(x_embed * target_x_embed, dim=3, keepdim=True) / (float(c_embed / num_temporal_attention_blocks)**0.5)
        ada_weights = ada_weights.expand(-1, -1, -1, int(c_embed / num_temporal_attention_blocks)).contiguous()
        ada_weights = ada_weights.view(batch_size, img_n, c_embed)
        ada_weights = ada_weights.softmax(dim=1)

        # Aggregation and generation of enhanced representation
        x = (x * ada_weights).sum(dim=1)
        del ada_weights
        x = x.view(batch_size, 1, s, 1)  # Size becomes [batch_size, 1, s, 1]
        # Define the upsampling layer
        upsample = nn.UpsamplingBilinear2d((orig_x1.size()[-2], orig_x1.size()[-1])).to(device)
        # Upsample x
        x_upsampled = upsample(x)
        # Remove the channel dimension
        x_upsampled = x_upsampled.squeeze(1)
        aggregated_enhanced_representation = orig_x1 + x_upsampled
        del x_upsampled
        bx1=aggregated_enhanced_representation
        batch_size, bl1, s = bx1.size(0), bx1.size(1), bx1.size(2)
        del aggregated_enhanced_representation
    
        linear_layer_back = nn.Linear(64, 256).to(device)
        bx1 = bx1.view(-1, s)  # Reshape to (4*30, 256)
        bx1 = linear_layer_back(bx1)  # Apply the linear layer
        bx1 = bx1.view(batch_size, bl1, 256)

        print("Size of bx1:", bx1.shape)
        
        
        
        bx1_flattened = bx1.view(bx1.size(0), -1)
        linear_layer = nn.Linear(bx1_flattened.size(1), input_query_label_2.size(1) * bx1.size(2)).to(bx1.device)
        lx1_transformed = linear_layer(bx1_flattened)
        lx1 = lx1_transformed.view(bx1.size(0), input_query_label_2.size(1), bx1.size(2))
        

        
        print("before free:",torch.cuda.memory_allocated(device))
        await asyncio.sleep(10)
        '''
        tensor_names = ['x1', 'x2', 'orig_x1', 'x2_flattened', 'x2_transformed', 'x2_final', 'x', 'x_embed', 'target_x_embed', 'bx1', 'bx1_flattened', 'lx1_transformed']
        num_iterations=100
        # Delete each tensor
        for i in range(num_iterations):
            for name in tensor_names:
                   del locals()[name]
                   torch.cuda.empty_cache()
                   gc.collect()
            #for obj in gc.get_objects():
                #if torch.is_tensor(obj):
                #   print(obj.size(), obj.device, obj.dtype)
        print("after free:",torch.cuda.memory_allocated(device))
        '''
        return lx1
async def main(): 
        # Sample tensors
        x1 = torch.randn(4, 30, 256)
        x2 = torch.randn(4, 300, 256)
        
        # Call the function
        result = await enhanced_representation(x1, x2)
        print(f"aggregated_enhanced_representation: {result.size()}")

# Run the main asynchronous function
asyncio.run(main())


