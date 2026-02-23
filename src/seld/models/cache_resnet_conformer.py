import torch
import torch.nn as nn

from typing import List

from seld.models.components.cache_resnet import resnet18_nopool
from seld.models.components.cache_conformer import ConformerBlock
from seld.models.components.resnet_conformer_audio import ResnetConformer_sed_doa_nopool_original, ResnetConformer_sed_doa_nopool_return_conformer_outputs

# from cache_resnet import resnet18_nopool
# from cache_conformer import ConformerBlock
import pdb
from seld.models.components.mask import add_optional_chunk_mask

class ResnetConformer_sed_doa_nopool(nn.Module):
    def __init__(self, in_channel, in_dim, out_dim, 
                 att_context_size:List[int] = [100,49], 
                 num_conformer_layer = 8,
                 encoder_dim = 256): # 7,64,39,[100,49],8,256
        super().__init__()
        self.resnet = resnet18_nopool(in_channel=in_channel)
        embedding_dim = in_dim // 32 * 256
        self.encoder_dim = encoder_dim
        self.in_ch = in_channel
        self.in_dim = in_dim
        self.att_context_size = att_context_size
        self.cache_past_len = self.att_context_size[0]
        self.input_projection = nn.Sequential(
            nn.Linear(embedding_dim, self.encoder_dim),
            nn.Dropout(p=0.05),
        )
        num_layers = num_conformer_layer
        self.conformer_layers = nn.ModuleList(
            [ConformerBlock(
                dim = self.encoder_dim,
                dim_head = 32,
                heads = 8,
                ff_mult = 2,
                conv_expansion_factor = 2,
                conv_kernel_size = 7,
                attn_dropout = 0.1,
                ff_dropout = 0.1,
                conv_dropout = 0.1,
                att_context_size = att_context_size
            ) for _ in range(num_layers)]
        )
        self.t_pooling = nn.MaxPool1d(kernel_size=5)
        self.sed_out_layer = nn.Sequential(
            nn.Linear(self.encoder_dim, self.encoder_dim),
            nn.LeakyReLU(),
            nn.Linear(self.encoder_dim, 13),
            nn.Sigmoid()
        )
        self.out_layer = nn.Sequential(
            nn.Linear(self.encoder_dim, self.encoder_dim),
            nn.LeakyReLU(),
            nn.Linear(self.encoder_dim, out_dim),
            nn.Tanh()
        ) 
        
    # def forward(self, x, resnet_cache=None, conformer_cache=None):
    #     if resnet_cache is None and conformer_cache is None:
    #     # if cache is None:
    #         conv_outputs = self.resnet(x)
    #         N, C, T, W = conv_outputs.shape
    #         conv_outputs = conv_outputs.permute(0,2,1,3).reshape(N, T, C*W)
            
    #         conformer_outputs = self.input_projection(conv_outputs)
            
    #         # === 新增：生成 Optional Mask ===
    #         # 1. 创建 padding mask (如果输入是固定长度或没有 padding 信息，全为 True)
    #         # 如果你有真实的长度信息 lengths，应该使用 make_pad_mask(lengths)
    #         masks = torch.ones(N, 1, T, dtype=torch.bool, device=conformer_outputs.device)
            
    #         # 2. 计算 chunk 参数 (沿用原有的 att_context_size 配置)
    #         chunk_size = self.att_context_size[1] + 1
    #         num_left_chunks = self.att_context_size[0] // chunk_size if chunk_size > 0 else -1
            
    #         # 3. 调用 add_optional_chunk_mask
    #         # use_dynamic_chunk=True 开启随机动态 chunk 训练 (推荐用于增强鲁棒性)
    #         # 如果想保持和原来完全一致的静态 chunk，设置 use_dynamic_chunk=False, static_chunk_size=chunk_size
    #         chunk_mask = add_optional_chunk_mask(
    #             conformer_outputs,
    #             masks,
    #             use_dynamic_chunk=True,  # 或 False，取决于你想是否要在训练中随机改变 chunk 大小
    #             use_dynamic_left_chunk=True,
    #             decoding_chunk_size=0,   # 0 表示训练时使用随机 chunk
    #             static_chunk_size=chunk_size, 
    #             num_decoding_left_chunks=num_left_chunks
    #         )
            
    #         # 4. 调整维度以适配 Multi-head Attention: (B, L, L) -> (B, 1, L, L)
    #         if chunk_mask is not None:
    #             chunk_mask = chunk_mask.unsqueeze(1)
            
    #         for layer in self.conformer_layers:
    #             conformer_outputs = layer(conformer_outputs)
    def forward(self, x, resnet_cache=None, conformer_cache=None, decoding_chunk_size=None):
        if resnet_cache is None and conformer_cache is None:
            # === 非流式模式 (训练 或 离线推理) ===
            
            conv_outputs = self.resnet(x)
            N, C, T, W = conv_outputs.shape
            conv_outputs = conv_outputs.permute(0,2,1,3).reshape(N, T, C*W)
            conformer_outputs = self.input_projection(conv_outputs)
            
            # === 核心修改逻辑 ===
            # 1. 确定 decoding_chunk_size
            if decoding_chunk_size is None:
                if self.training:
                    # 训练模式：使用随机动态 Chunk (0)
                    active_decoding_chunk_size = 0
                else:
                    # 推理模式：默认使用全上下文 (-1)，获得最佳性能
                    # 如果你想在验证时模拟流式模型的固定延迟，这里可以改成 self.att_context_size[1] + 1
                    active_decoding_chunk_size = self.att_context_size[1] + 1
            # 2. 生成 mask
            masks = torch.ones(N, 1, T, dtype=torch.bool, device=conformer_outputs.device)
            chunk_size = self.att_context_size[1] + 1
            num_left_chunks = self.att_context_size[0] // chunk_size if chunk_size > 0 else -1

            chunk_mask = add_optional_chunk_mask(
                conformer_outputs,
                masks,
                use_dynamic_chunk=True,  # 必须开启这个才能让 active_decoding_chunk_size 生效
                use_dynamic_left_chunk=True,
                decoding_chunk_size=active_decoding_chunk_size, # 传入决定好的 size
                static_chunk_size=chunk_size, 
                num_decoding_left_chunks=num_left_chunks
            )
            
            if chunk_mask is not None:
                chunk_mask = chunk_mask.unsqueeze(1)
            # ===================

            for layer in self.conformer_layers:
                conformer_outputs = layer(conformer_outputs, mask=chunk_mask)      
            outputs = conformer_outputs.permute(0,2,1)
            outputs = self.t_pooling(outputs)
            outputs = outputs.permute(0,2,1)
            
            sed = self.sed_out_layer(outputs)
            doa = self.out_layer(outputs)
            pred = torch.cat((sed, doa), dim=-1)
            return pred
        else:
            layer_caches = conformer_cache
            # Streaming inference mode
            conv_outputs, next_resnet_cache = self.resnet(x, resnet_cache)
            N,C,T,W = conv_outputs.shape
            conv_outputs = conv_outputs.permute(0,2,1,3).reshape(N, T, C*W)
            
            conformer_outputs = self.input_projection(conv_outputs)
            
            next_layer_caches = []
            for i, layer in enumerate(self.conformer_layers):
                conformer_outputs, next_cache = layer(
                    conformer_outputs,
                    cache=layer_caches[i] if layer_caches else None
                )
                next_layer_caches.append(next_cache)
                
            outputs = conformer_outputs.permute(0,2,1)
            outputs = self.t_pooling(outputs)
            outputs = outputs.permute(0,2,1)
            
            sed = self.sed_out_layer(outputs)
            doa = self.out_layer(outputs)
            pred = torch.cat((sed, doa), dim=-1)
            
            return pred, (next_resnet_cache, next_layer_caches)

    def get_initial_cache_resnet(self, batch_size=1):
        """初始化ResNet的卷积缓存"""
        device = next(self.parameters()).device
        
        conv1_cache = torch.zeros(batch_size, self.in_ch, 2, self.in_dim, device=device)
        
        layer_caches = []
        channels = [(24, 24), (48, 48), (96, 96), (192, 192)]
        features = [64, 16, 4, 2]
        
        for layer_idx, ((in_ch, out_ch), feat_dim) in enumerate(zip(channels, features)):
            current_layer_caches = []  # 当前layer的所有cache
            num_blocks = 2
            
            for block_idx in range(num_blocks):
                # 为每个block创建两个cache
                if block_idx == 0 and layer_idx > 0:
                    prev_ch = channels[layer_idx-1][1]
                    cache1 = torch.zeros(batch_size, prev_ch, 2, feat_dim, device=device)
                else:
                    cache1 = torch.zeros(batch_size, in_ch, 2, feat_dim, device=device)
                cache2 = torch.zeros(batch_size, out_ch, 2, feat_dim, device=device)
                current_layer_caches.extend([cache1, cache2])
                
            layer_caches.append(current_layer_caches)  # 将当前layer的所有cache作为一个整体添加
        
        return (conv1_cache, layer_caches)

    def get_initial_cache_conformer(self, batch_size=1):
        caches = []
        device = next(self.parameters()).device  # 确保缓存在正确设备上
        for layer in self.conformer_layers:
            # attention缓存维度: [B, cache_len, D=256]
            attn_cache = torch.zeros(batch_size, self.cache_past_len, self.encoder_dim, device=device)
            
            # convolution缓存维度: [B, D=256, kernel_size-1=6]
            conv_cache = torch.zeros(batch_size, 2*self.encoder_dim, 6, device=device)
            
            caches.append((attn_cache, conv_cache))
        return caches
    
class ResnetConformer_sed_doa_nopool_TS(nn.Module):
    def __init__(self, in_channel, in_dim, out_dim, 
                 att_context_size = [100,49], 
                 num_conformer_layer = 8,
                 encoder_dim = 256): # 7,64,39,[100,49],8,256
        super().__init__()
        self.resnet = resnet18_nopool(in_channel=in_channel)
        self.teacher_model = ResnetConformer_sed_doa_nopool_original(in_channel=7, in_dim=64, out_dim=39)
        embedding_dim = in_dim // 32 * 256
        self.encoder_dim = encoder_dim
        self.in_ch = in_channel
        self.in_dim = in_dim
        self.att_context_size = att_context_size
        self.cache_past_len = self.att_context_size[0]
        self.input_projection = nn.Sequential(
            nn.Linear(embedding_dim, self.encoder_dim),
            nn.Dropout(p=0.05),
        )
        num_layers = num_conformer_layer
        self.conformer_layers = nn.ModuleList(
            [ConformerBlock(
                dim = self.encoder_dim,
                dim_head = 32,
                heads = 8,
                ff_mult = 2,
                conv_expansion_factor = 2,
                conv_kernel_size = 7,
                attn_dropout = 0.1,
                ff_dropout = 0.1,
                conv_dropout = 0.1,
                att_context_size = att_context_size
            ) for _ in range(num_layers)]
        )
        self.t_pooling = nn.MaxPool1d(kernel_size=5)
        self.sed_out_layer = nn.Sequential(
            nn.Linear(self.encoder_dim, self.encoder_dim),
            nn.LeakyReLU(),
            nn.Linear(self.encoder_dim, 13),
            nn.Sigmoid()
        )
        self.out_layer = nn.Sequential(
            nn.Linear(self.encoder_dim, self.encoder_dim),
            nn.LeakyReLU(),
            nn.Linear(self.encoder_dim, out_dim),
            nn.Tanh()
        ) 
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    def forward(self, x, resnet_cache=None, conformer_cache=None):
        if resnet_cache is None and conformer_cache is None:
        # if cache is None:
            target_ts = self.teacher_model(x)
            conv_outputs = self.resnet(x)
            N, C, T, W = conv_outputs.shape
            conv_outputs = conv_outputs.permute(0,2,1,3).reshape(N, T, C*W)
            
            conformer_outputs = self.input_projection(conv_outputs)
            
            for layer in self.conformer_layers:
                conformer_outputs = layer(conformer_outputs)
                
            outputs = conformer_outputs.permute(0,2,1)
            outputs = self.t_pooling(outputs)
            outputs = outputs.permute(0,2,1)
            
            sed = self.sed_out_layer(outputs)
            doa = self.out_layer(outputs)
            pred = torch.cat((sed, doa), dim=-1)
            return pred, target_ts
        else:
            target_ts = self.teacher_model(x)
            layer_caches = conformer_cache
            # Streaming inference mode
            conv_outputs, next_resnet_cache = self.resnet(x, resnet_cache)
            N,C,T,W = conv_outputs.shape
            conv_outputs = conv_outputs.permute(0,2,1,3).reshape(N, T, C*W)
            
            conformer_outputs = self.input_projection(conv_outputs)
            
            next_layer_caches = []
            for i, layer in enumerate(self.conformer_layers):
                conformer_outputs, next_cache = layer(
                    conformer_outputs,
                    cache=layer_caches[i] if layer_caches else None
                )
                next_layer_caches.append(next_cache)
                
            outputs = conformer_outputs.permute(0,2,1)
            outputs = self.t_pooling(outputs)
            outputs = outputs.permute(0,2,1)
            
            sed = self.sed_out_layer(outputs)
            doa = self.out_layer(outputs)
            pred = torch.cat((sed, doa), dim=-1)
            
            return pred, target_ts, (next_resnet_cache, next_layer_caches)

    def get_initial_cache_resnet(self, batch_size=1):
        """初始化ResNet的卷积缓存"""
        device = next(self.parameters()).device
        
        conv1_cache = torch.zeros(batch_size, self.in_ch, 2, self.in_dim, device=device)
        
        layer_caches = []
        channels = [(24, 24), (48, 48), (96, 96), (192, 192)]
        features = [64, 16, 4, 2]
        
        for layer_idx, ((in_ch, out_ch), feat_dim) in enumerate(zip(channels, features)):
            current_layer_caches = []  # 当前layer的所有cache
            num_blocks = 2
            
            for block_idx in range(num_blocks):
                # 为每个block创建两个cache
                if block_idx == 0 and layer_idx > 0:
                    prev_ch = channels[layer_idx-1][1]
                    cache1 = torch.zeros(batch_size, prev_ch, 2, feat_dim, device=device)
                else:
                    cache1 = torch.zeros(batch_size, in_ch, 2, feat_dim, device=device)
                cache2 = torch.zeros(batch_size, out_ch, 2, feat_dim, device=device)
                current_layer_caches.extend([cache1, cache2])
                
            layer_caches.append(current_layer_caches)  # 将当前layer的所有cache作为一个整体添加
        
        return (conv1_cache, layer_caches)

    def get_initial_cache_conformer(self, batch_size=1):
        caches = []
        device = next(self.parameters()).device  # 确保缓存在正确设备上
        for layer in self.conformer_layers:
            # attention缓存维度: [B, cache_len, D=256]
            attn_cache = torch.zeros(batch_size, self.cache_past_len, self.encoder_dim, device=device)
            
            # convolution缓存维度: [B, D=256, kernel_size-1=6]
            conv_cache = torch.zeros(batch_size, 2*self.encoder_dim, 6, device=device)
            
            caches.append((attn_cache, conv_cache))
        return caches
    

class ResnetConformer_sed_doa_nopool_TS_after_conformer(nn.Module):
    def __init__(self, in_channel, in_dim, out_dim, 
                 att_context_size = [100,49], 
                 num_conformer_layer = 8,
                 encoder_dim = 256): # 7,64,39,[100,49],8,256
        super().__init__()
        self.resnet = resnet18_nopool(in_channel=in_channel)
        self.teacher_model = ResnetConformer_sed_doa_nopool_return_conformer_outputs(in_channel=7, in_dim=64, out_dim=39)
        embedding_dim = in_dim // 32 * 256
        self.encoder_dim = encoder_dim
        self.in_ch = in_channel
        self.in_dim = in_dim
        self.att_context_size = att_context_size
        self.cache_past_len = self.att_context_size[0]
        self.input_projection = nn.Sequential(
            nn.Linear(embedding_dim, self.encoder_dim),
            nn.Dropout(p=0.05),
        )
        num_layers = num_conformer_layer
        self.conformer_layers = nn.ModuleList(
            [ConformerBlock(
                dim = self.encoder_dim,
                dim_head = 32,
                heads = 8,
                ff_mult = 2,
                conv_expansion_factor = 2,
                conv_kernel_size = 7,
                attn_dropout = 0.1,
                ff_dropout = 0.1,
                conv_dropout = 0.1,
                att_context_size = att_context_size
            ) for _ in range(num_layers)]
        )
        self.t_pooling = nn.MaxPool1d(kernel_size=5)
        self.sed_out_layer = nn.Sequential(
            nn.Linear(self.encoder_dim, self.encoder_dim),
            nn.LeakyReLU(),
            nn.Linear(self.encoder_dim, 13),
            nn.Sigmoid()
        )
        self.out_layer = nn.Sequential(
            nn.Linear(self.encoder_dim, self.encoder_dim),
            nn.LeakyReLU(),
            nn.Linear(self.encoder_dim, out_dim),
            nn.Tanh()
        ) 
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='mean')
        
        # 修改教师模型以便返回中间表示
        self.teacher_model.return_conformer_output = True
        
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
    def forward(self, x, resnet_cache=None, conformer_cache=None):
        if resnet_cache is None and conformer_cache is None:
            # 获取教师模型的输出和中间表示
            target_ts, teacher_conformer_output = self.teacher_model(x)
            
            # 学生模型前向传播
            conv_outputs = self.resnet(x)
            N, C, T, W = conv_outputs.shape
            conv_outputs = conv_outputs.permute(0,2,1,3).reshape(N, T, C*W)
            
            conformer_outputs = self.input_projection(conv_outputs)
            
            for layer in self.conformer_layers:
                conformer_outputs = layer(conformer_outputs)
            
            # 计算教师和学生的Conformer输出之间的Smooth L1 Loss
            representation_loss = self.smooth_l1_loss(conformer_outputs, teacher_conformer_output)
            
            # 继续前向传播完成预测
            outputs = conformer_outputs.permute(0,2,1)
            outputs = self.t_pooling(outputs)
            outputs = outputs.permute(0,2,1)
            
            sed = self.sed_out_layer(outputs)
            doa = self.out_layer(outputs)
            pred = torch.cat((sed, doa), dim=-1)
            
            return pred, target_ts, representation_loss
        else:
            # 流式推理模式
            target_ts, teacher_conformer_output = self.teacher_model(x)
            layer_caches = conformer_cache
            
            conv_outputs, next_resnet_cache = self.resnet(x, resnet_cache)
            N,C,T,W = conv_outputs.shape
            conv_outputs = conv_outputs.permute(0,2,1,3).reshape(N, T, C*W)
            
            conformer_outputs = self.input_projection(conv_outputs)
            
            next_layer_caches = []
            for i, layer in enumerate(self.conformer_layers):
                conformer_outputs, next_cache = layer(
                    conformer_outputs,
                    cache=layer_caches[i] if layer_caches else None
                )
                next_layer_caches.append(next_cache)
            
            # 计算教师和学生的Conformer输出之间的Smooth L1 Loss
            representation_loss = self.smooth_l1_loss(conformer_outputs, teacher_conformer_output)
                
            outputs = conformer_outputs.permute(0,2,1)
            outputs = self.t_pooling(outputs)
            outputs = outputs.permute(0,2,1)
            
            sed = self.sed_out_layer(outputs)
            doa = self.out_layer(outputs)
            pred = torch.cat((sed, doa), dim=-1)
            
            return pred, target_ts, representation_loss, (next_resnet_cache, next_layer_caches)

    def get_initial_cache_resnet(self, batch_size=1):
        """初始化ResNet的卷积缓存"""
        device = next(self.parameters()).device
        
        conv1_cache = torch.zeros(batch_size, self.in_ch, 2, self.in_dim, device=device)
        
        layer_caches = []
        channels = [(24, 24), (48, 48), (96, 96), (192, 192)]
        features = [64, 16, 4, 2]
        
        for layer_idx, ((in_ch, out_ch), feat_dim) in enumerate(zip(channels, features)):
            current_layer_caches = []  # 当前layer的所有cache
            num_blocks = 2
            
            for block_idx in range(num_blocks):
                # 为每个block创建两个cache
                if block_idx == 0 and layer_idx > 0:
                    prev_ch = channels[layer_idx-1][1]
                    cache1 = torch.zeros(batch_size, prev_ch, 2, feat_dim, device=device)
                else:
                    cache1 = torch.zeros(batch_size, in_ch, 2, feat_dim, device=device)
                cache2 = torch.zeros(batch_size, out_ch, 2, feat_dim, device=device)
                current_layer_caches.extend([cache1, cache2])
                
            layer_caches.append(current_layer_caches)  # 将当前layer的所有cache作为一个整体添加
        
        return (conv1_cache, layer_caches)

    def get_initial_cache_conformer(self, batch_size=1):
        caches = []
        device = next(self.parameters()).device  # 确保缓存在正确设备上
        for layer in self.conformer_layers:
            # attention缓存维度: [B, cache_len, D=256]
            attn_cache = torch.zeros(batch_size, self.cache_past_len, self.encoder_dim, device=device)
            
            # convolution缓存维度: [B, D=256, kernel_size-1=6]
            conv_cache = torch.zeros(batch_size, 2*self.encoder_dim, 6, device=device)
            
            caches.append((attn_cache, conv_cache))
        return caches

# if __name__ == "__main__":
#     # 模型初始化
#     att_context_size=[100,24]
#     model = ResnetConformer_sed_doa_nopool(in_channel=7, in_dim=64, out_dim=39, 
#                                            att_context_size=att_context_size,
#                                            num_conformer_layer = 8,
#                                            encoder_dim = 128)
#     model.eval()  # 设置为评估模式

#     # 创建模拟输入
#     batch_size = 1
#     input_tensor = torch.randn(batch_size, 7, 690, 64)
    
#     # 流式推理参数
#     chunk_size = att_context_size[1] + 1
    
#     # 初始化缓存
#     resnet_cache = model.get_initial_cache_resnet(batch_size)
#     conformer_cache = model.get_initial_cache_conformer(batch_size)
    
#     # 存储所有输出用于对比
#     streaming_outputs = []
    
#     # 流式推理
#     print("开始流式推理...")
#     for i in range(0, 690, chunk_size):
#         chunk = input_tensor[:, :, i:i+chunk_size, :]
        
#         # 打印当前处理的时间步
#         print(f"Processing frames {i} to {i+chunk_size}")
        
#         # 如果是第一个块
#         if i == 0:
#             output, (next_resnet_cache, next_conformer_cache) = model(
#                 chunk, 
#                 resnet_cache=resnet_cache, 
#                 conformer_cache=conformer_cache
#             )
#         else:
#             output, (next_resnet_cache, next_conformer_cache) = model(
#                 chunk,
#                 resnet_cache=next_resnet_cache,
#                 conformer_cache=next_conformer_cache
#             )
        
#         # 存储当前块的输出
#         streaming_outputs.append(output)
        
#         # 打印当前输出形状和缓存信息
#         print(f"Output shape: {output.shape}")
#         print(f"First conformer layer attention cache shape: {next_conformer_cache[0][0].shape}")
#         print(f"First conformer layer conv cache shape: {next_conformer_cache[0][1].shape}")
#         print("---")
    
#     # 将所有输出拼接在一起
#     streaming_output = torch.cat(streaming_outputs, dim=1)
#     print(f"Final streaming output shape: {streaming_output.shape}")
    
#     # 对比非流式处理的结果
#     print("\n验证非流式处理...")
#     with torch.no_grad():
#         non_streaming_output = model(input_tensor)
#     print(f"Non-streaming output shape: {non_streaming_output.shape}")
    
#     # 比较两种处理方式的输出差异
#     if streaming_output.shape == non_streaming_output.shape:
#         max_diff = torch.max(torch.abs(streaming_output - non_streaming_output))
#         print(f"\nMaximum difference between streaming and non-streaming outputs: {max_diff}")
#     else:
#         print("\nWarning: Output shapes don't match!")
#         print(f"Streaming shape: {streaming_output.shape}")
#         print(f"Non-streaming shape: {non_streaming_output.shape}")