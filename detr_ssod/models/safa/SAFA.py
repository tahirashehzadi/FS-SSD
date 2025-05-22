class ScaleAwareFeatureAggregation(nn.Module):

 

    """
      ScaleAwareFeatureAggregation happens in five steps

 

      1 : Downgrade the size of two feature maps.
      2 : Converts the feature maps into query and key representaions.
      3 : Concats the downgraded features and divide it into blocks.
      4 : Performs Multi-Head Attention.
      5 : Aggregates and outputs the enhanced representation

 

    """

 

 

    def __init__(self, channels, query_image_size, key_image_size):
        super().__init__()

 

        query_stride = query_image_size // 79
        key_stride = key_image_size // 79
        key_kernel = key_stride
        if key_stride > 1:
            key_kernel = 3

 

        self.num_temporal_attention_blocks = 8
        if self.num_temporal_attention_blocks  > 0:
            self.query_conv1 = nn.Conv2d(in_channels = channels , out_channels = channels , kernel_size= 7, stride=4)
            self.query_conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2)
            self.key_conv = nn.Conv2d(in_channels = channels , out_channels = channels , kernel_size=3, stride=2)

 

 

    def forward(self, x, quarter_scale_x):
        """ 
        Args : 
            x:  a list of, feature map in (C, H, W) format.
            quarter_scale_x: a list of, quarter scale feature map in (C, H/4, W/4) format.

 

        Returns :
            aggregated_enhanced_representation : enhanced aggregated feature representations of
                                                 two feature maps from different scales.
                                                 a list of, enhanced features in (C, H, W)

 

        """

 

        orig_x = x

 

        # Key Query Generation
        #x = self.query_conv1(x)
        #x = self.query_conv2(x)
        #quarter_scale_x = self.key_conv(quarter_scale_x)
        orig_x = x
        self.num_temporal_attention_blocks = 8
        batch_size, C, roi_h, roi_w = x.size()

        x = x.view(batch_size, 1, C, roi_h, roi_w)
        quarter_scale_x = quarter_scale_x.view(batch_size, 1, C, roi_h, roi_w)

 

        x = torch.cat((x, quarter_scale_x), dim=1)
        batch_size, img_n, _, roi_h, roi_w = x.size()

 

        # Calculating the number of attention blocks
        num_attention_blocks = self.num_temporal_attention_blocks
        x_embed = x
        c_embed = x_embed.size(2)

 

        # Performing multi-head attention
        # (img_n, num_attention_blocks, C / num_attention_blocks, H, W)
        x_embed = x_embed.view(batch_size, img_n, num_attention_blocks, -1, roi_h,
                               roi_w)
        # (1, roi_n, num_attention_blocks, C / num_attention_blocks, H, W)
        target_x_embed = x_embed[:, [1]]
        # (batch_size, img_n, num_attention_blocks, 1, H, W)
        ada_weights = torch.sum(
            x_embed * target_x_embed, dim=3, keepdim=True) / (
                float(c_embed / num_attention_blocks)**0.5)
        # (batch_size, img_n, num_attention_blocks, C / num_attention_blocks, H, W)
        ada_weights = ada_weights.expand(-1, -1, -1,
                                         int(c_embed / num_attention_blocks),
                                         -1, -1).contiguous()
        ada_weights = ada_weights.view(batch_size, img_n, c_embed, roi_h, roi_w)
        ada_weights = ada_weights.softmax(dim=1)

 

        # Aggregation and generation of enhanced representation
        x = (x * ada_weights).sum(dim=1)
        upsample = nn.UpsamplingBilinear2d((orig_x.size()[-2], orig_x.size()[-1]))
        aggregated_feature = upsample(x)
        aggregated_enhanced_representation = orig_x + aggregated_feature
        return aggregated_enhanced_representation