import torch
import torch.nn as nn
import math
from .kan import AttentionWithFastKANTransform
import torch.nn.functional as F


class eca_bloack(nn.Module):#Efficient Channel Attention
    def __init__(self, channel, gamma=2, b=1):#，用于生成基于通道的权重，通过逐元素乘法增强重要的特征通道
        super(eca_bloack, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.ave_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        #进行特征提取，最后用Sigmoid函数生成通道注意力权重
    def forward(self, x):
        b, c, l = x.size()
        avg = self.ave_pool(x).view([b, 1, c])
        out = self.conv(avg)
        out = self.sigmoid(out).view([b, c, 1])
        return out * x


class GC(torch.nn.Module):#基于全局上下文，用于捕捉特征间的全局关系
    def __init__(self, in_channel, ratio):
        super(GC, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channel, 1, kernel_size=1)
        self.conv2 = torch.nn.Conv1d(in_channel, in_channel // ratio, kernel_size=1)
        self.conv3 = torch.nn.Conv1d(in_channel // ratio, in_channel, kernel_size=1)
        self.softmax = torch.nn.Softmax(dim=1)
        self.ln = torch.nn.LayerNorm([in_channel // ratio, 1])
        self.relu = torch.nn.ReLU()

    def forward(self, input):
        b, c, w = input.shape
        x = self.conv1(input).permute(0, 2, 1)
        x = self.softmax(x)
        i = input.contiguous()
        x = torch.bmm(i, x).view([b, c, 1])
        x = self.conv2(x)
        x = self.ln(x)
        x = self.relu(x)
        x = self.conv3(x)

        return x + input


# 获得多个卷积层，卷积核大小分别为1, 3, 5
class convlayer(torch.nn.Sequential):
    def __init__(self, in_channel=1):
        super(convlayer, self).__init__()
        kernel_sizes = [1, 3, 5]
        for i, kernel_size in enumerate(kernel_sizes):
            layer = torch.nn.Conv1d(in_channel, in_channel, kernel_size=kernel_size, padding=kernel_size // 2)
            self.add_module('convlayer%d' % i, layer)

# 获得layer_num=3个用于反向压缩卷积的线性层
class linearlayer(torch.nn.Sequential):
    def __init__(self, in_channel, out_channel, layer_num=3):
        super(linearlayer, self).__init__()
        for i in range(layer_num):
            layer = torch.nn.Linear(in_channel, out_channel)
            self.add_module('linearlayer%d' % i, layer)

class SK(torch.nn.Module):# 选择性卷积模块（Selective Kernel），
    def __init__(self, in_channel, layer_num):
        super(SK, self).__init__()
        self.conv = convlayer(in_channel)
        self.linear1 = torch.nn.Linear(in_channel, in_channel)
        self.linear2 = linearlayer(in_channel, in_channel, layer_num)
        self.softmax = torch.nn.Softmax(dim=1)
        self.ave = torch.nn.AdaptiveAvgPool1d(1)

    def forward(self, input):
        b, c, w = input.shape
        x = torch.zeros([b, c, w], device=input.device)
        x_list = []

        for i in self.conv:
            res = i(input)
            x_list.append(res)
            x += res

        x = self.ave(x)
        x = x.view([b, c])
        x = self.linear1(x)
        # 通过多尺度卷积和通道自适应选择机制来动态调整特征的尺度
        output = torch.zeros([b, c, w], device=input.device)

        for j, k in enumerate(self.linear2):
            s = k(x)
            s = s.view([b, c, 1])
            s = self.softmax(s)
            output += s * x_list[j]

        return output

class PositionEmbeddingBlock(nn.Module):#位置嵌入（Position Embedding），用于为序列数据提供位置信息
    def __init__(self, d_model, max_len=1):
        super(PositionEmbeddingBlock, self).__init__()
        self.pe = torch.zeros(max_len, d_model, device='cuda' if torch.cuda.is_available() else 'cpu')
        position = torch.arange(0, max_len, dtype=torch.float, device=self.pe.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=self.pe.device) * (
                -torch.log(torch.tensor(10000.0, device=self.pe.device)) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)  # 增加 batch 维度

    def forward(self, x):
        return self.pe[:, :x.size(1)]  # 仅使用输入序列长度范围内的位置编码


class MultiScaleAttentionTransformer(nn.Module):#多尺度注意力Transformer
    def __init__(self, embed_size, num_heads, num_layers, forward_expansion, dropout, output_dim):
        super(MultiScaleAttentionTransformer, self).__init__()

        #卷特征提取层
        #三种卷积核大小（1、3、5）的卷积层（conv1、conv3、conv5），以提取多尺度的特征
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, padding=2)

        #位置嵌入层
        self.pos_embedding = PositionEmbeddingBlock(embed_size)


        #全连接层
        self.fc1 = nn.Linear(8, embed_size)
        self.fc3 = nn.Linear(8, embed_size)
        self.fc5 = nn.Linear(8, embed_size)

        #多头注意力层，不同卷积尺度特征
        self.multihead_attn1 = nn.MultiheadAttention(embed_size, num_heads) # 64,8
        self.multihead_attn2 = nn.MultiheadAttention(embed_size, num_heads)

        #TransformerEncoder，编码层，处理不同尺度的卷积输出
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads,
                                                   dim_feedforward=forward_expansion * embed_size, dropout=dropout)
        self.transformer_encoder_1 = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.transformer_encoder_3 = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.transformer_encoder_5 = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        #特征融合，将三个尺度的输出特征在通道维度上拼接，归一化（norm），线性层（fc_out）进一步融合
        self.norm = nn.LayerNorm(embed_size * 3)
        self.fc_out = nn.Linear(embed_size * 3, embed_size)
        self.dropout = nn.Dropout(dropout)


        # Decoupled KAN 模块。将多尺度融合特征解耦，该模块将多尺度特征解耦
        self.kan_transform = AttentionWithFastKANTransform(
            q_dim=embed_size,  # 使用 fc_out 的输出作为查询维度
            k_dim=embed_size,
            v_dim=embed_size,
            head_dim=embed_size // num_heads,
            num_heads=num_heads
        )
        # 添加一个线性层，将输出维度从 64 映射到 3
        self.final_fc = nn.Linear(embed_size, output_dim)


    def forward(self, x):
        x = x.unsqueeze(1)

        conv1_out = self.conv1(x)
        conv3_out = self.conv3(x)
        conv5_out = self.conv5(x)


        conv1_out = self.fc1(conv1_out)
        conv3_out = self.fc3(conv3_out)
        conv5_out = self.fc5(conv5_out)

        conv1_out = self.pos_embedding(conv1_out) + conv1_out
        conv3_out = self.pos_embedding(conv3_out) + conv3_out
        conv5_out = self.pos_embedding(conv5_out) + conv5_out

        attn_output3, _ = self.multihead_attn1(conv1_out, conv3_out, conv3_out)
        attn_output5, _ = self.multihead_attn2(conv1_out, conv5_out, conv5_out)

        out1 = self.transformer_encoder_1(conv1_out)
        out3 = self.transformer_encoder_3(attn_output3)
        out5 = self.transformer_encoder_5(attn_output5)

        combined = torch.cat((out1, out3, out5), dim=2)
        combined = self.norm(combined)
        combined = self.fc_out(combined)
        combined = combined.mean(dim=1)

        # 使用 AttentionWithFastKANTransform 进行最终的解耦和多向预测
        #out = self.kan_transform(combined, combined, combined)
        out = self.final_fc(combined)
        return out

#设计消融实验模型
#实验组1，移除多尺度卷积
class AblationSingleScale(nn.Module):
    def __init__(self, embed_size, num_heads, num_layers, forward_expansion, dropout, output_dim):
        super(AblationSingleScale, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1)
        self.fc1 = nn.Linear(8, embed_size)
        self.pos_embedding = PositionEmbeddingBlock(embed_size)
        self.multihead_attn1 = nn.MultiheadAttention(embed_size, num_heads)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads,
                                                   dim_feedforward=forward_expansion * embed_size, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_size, embed_size)
        self.final_fc = nn.Linear(embed_size, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        conv1_out = self.conv1(x)
        conv1_out = self.fc1(conv1_out)
        conv1_out = self.pos_embedding(conv1_out) + conv1_out
        attn_output, _ = self.multihead_attn1(conv1_out, conv1_out, conv1_out)
        out = self.transformer_encoder(attn_output)
        out = out.mean(dim=1)
        out = self.fc_out(out)
        output = self.final_fc(out)
        return output
#实验组2，移除多尺度注意力
class AblationNoAttention(nn.Module):
    def __init__(self, embed_size, num_heads, num_layers, forward_expansion, dropout, output_dim):
        super(AblationNoAttention, self).__init__()

        # 卷特征提取层
        # 三种卷积核大小（1、3、5）的卷积层（conv1、conv3、conv5），以提取多尺度的特征
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, padding=2)

        # 位置嵌入层
        self.pos_embedding = PositionEmbeddingBlock(embed_size)

        #全连接层
        self.fc1 = nn.Linear(8, embed_size)
        self.fc3 = nn.Linear(8, embed_size)
        self.fc5 = nn.Linear(8, embed_size)

        # TransformerEncoder，编码层
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads,
                                                   dim_feedforward=forward_expansion * embed_size, dropout=dropout)
        self.transformer_encoder_1 = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.transformer_encoder_3 = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.transformer_encoder_5 = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(embed_size * 3)
        self.fc_out = nn.Linear(embed_size * 3, embed_size)
        self.dropout = nn.Dropout(dropout)

        # 添加一个线性层，将输出维度从 64 映射到 3
        self.final_fc = nn.Linear(embed_size, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)

        conv1_out = self.conv1(x)
        conv3_out = self.conv3(x)
        conv5_out = self.conv5(x)

        conv1_out = self.fc1(conv1_out)
        conv3_out = self.fc3(conv3_out)
        conv5_out = self.fc5(conv5_out)

        conv1_out = self.pos_embedding(conv1_out) + conv1_out
        conv3_out = self.pos_embedding(conv3_out) + conv3_out
        conv5_out = self.pos_embedding(conv5_out) + conv5_out


        out1 = self.transformer_encoder_1(conv1_out)
        out3 = self.transformer_encoder_3(conv3_out)
        out5 = self.transformer_encoder_5(conv5_out)

        combined = torch.cat((out1, out3, out5), dim=2)
        combined = self.norm(combined)
        combined = self.fc_out(combined)
        combined = combined.mean(dim=1)

        out = self.final_fc(combined)
        return out
#实验组3，移除位置编码
class AblationNoPositionEncoding(MultiScaleAttentionTransformer):
    def forward(self, x):
        x = x.unsqueeze(1)
        conv1_out = self.fc1(self.conv1(x))
        conv3_out = self.fc3(self.conv3(x))
        conv5_out = self.fc5(self.conv5(x))
        attn_output3, _ = self.multihead_attn1(conv1_out, conv3_out, conv3_out)
        attn_output5, _ = self.multihead_attn2(conv1_out, conv5_out, conv5_out)
        out1 = self.transformer_encoder_1(conv1_out)
        out3 = self.transformer_encoder_3(attn_output3)
        out5 = self.transformer_encoder_5(attn_output5)
        combined = torch.cat((out1, out3, out5), dim=2)
        combined = self.fc_out(combined.mean(dim=1))
        return self.final_fc(combined)
#实验组4，X方向解耦
class AblationSingleDirectionOutputX(MultiScaleAttentionTransformer):
    def __init__(self, embed_size, num_heads, num_layers, forward_expansion, dropout, output_dim=1):
        """
        只输出单方向（例如X方向）预测值。
        """
        super(AblationSingleDirectionOutputX, self).__init__(embed_size, num_heads, num_layers, forward_expansion, dropout, output_dim)
        self.final_fc = nn.Linear(embed_size, 1)  # 强制输出维度为1，表示单方向结果

    def forward(self, x):
        x = x.unsqueeze(1)

        # 多尺度卷积层
        conv1_out = self.conv1(x)
        conv3_out = self.conv3(x)
        conv5_out = self.conv5(x)

        # 全连接层映射
        conv1_out = self.fc1(conv1_out)
        conv3_out = self.fc3(conv3_out)
        conv5_out = self.fc5(conv5_out)

        # 添加位置编码
        conv1_out = self.pos_embedding(conv1_out) + conv1_out
        conv3_out = self.pos_embedding(conv3_out) + conv3_out
        conv5_out = self.pos_embedding(conv5_out) + conv5_out

        # 多头注意力处理
        attn_output3, _ = self.multihead_attn1(conv1_out, conv3_out, conv3_out)
        attn_output5, _ = self.multihead_attn2(conv1_out, conv5_out, conv5_out)

        # Transformer编码
        out1 = self.transformer_encoder_1(conv1_out)
        out3 = self.transformer_encoder_3(attn_output3)
        out5 = self.transformer_encoder_5(attn_output5)

        # 特征融合与解耦
        combined = torch.cat((out1, out3, out5), dim=2)
        combined = self.norm(combined)
        combined = self.fc_out(combined)
        combined = combined.mean(dim=1)

        # 输出单方向（X方向）载荷值
        return self.final_fc(combined)
#实验组5，Y方向解耦
class AblationSingleDirectionOutputY(MultiScaleAttentionTransformer):
    def __init__(self, embed_size, num_heads, num_layers, forward_expansion, dropout, output_dim=1):
        """
        只输出Y方向预测值。
        """
        super(AblationSingleDirectionOutputY, self).__init__(embed_size, num_heads, num_layers, forward_expansion, dropout, output_dim)
        self.final_fc = nn.Linear(embed_size, 1)  # 输出维度为1，表示Y方向

    def forward(self, x):
        x = x.unsqueeze(1)

        # 多尺度卷积层
        conv1_out = self.conv1(x)
        conv3_out = self.conv3(x)
        conv5_out = self.conv5(x)

        # 全连接层映射
        conv1_out = self.fc1(conv1_out)
        conv3_out = self.fc3(conv3_out)
        conv5_out = self.fc5(conv5_out)

        # 添加位置编码
        conv1_out = self.pos_embedding(conv1_out) + conv1_out
        conv3_out = self.pos_embedding(conv3_out) + conv3_out
        conv5_out = self.pos_embedding(conv5_out) + conv5_out

        # 多头注意力处理
        attn_output3, _ = self.multihead_attn1(conv1_out, conv3_out, conv3_out)
        attn_output5, _ = self.multihead_attn2(conv1_out, conv5_out, conv5_out)

        # Transformer编码
        out1 = self.transformer_encoder_1(conv1_out)
        out3 = self.transformer_encoder_3(attn_output3)
        out5 = self.transformer_encoder_5(attn_output5)

        # 特征融合与解耦
        combined = torch.cat((out1, out3, out5), dim=2)
        combined = self.norm(combined)
        combined = self.fc_out(combined)
        combined = combined.mean(dim=1)

        # 输出Y方向载荷值
        return self.final_fc(combined)
#实验组6，Z方向解耦
class AblationSingleDirectionOutputZ(MultiScaleAttentionTransformer):
    def __init__(self, embed_size, num_heads, num_layers, forward_expansion, dropout, output_dim=1):
        """
        只输出Z方向预测值。
        """
        super(AblationSingleDirectionOutputZ, self).__init__(embed_size, num_heads, num_layers, forward_expansion, dropout, output_dim)
        self.final_fc = nn.Linear(embed_size, 1)  # 输出维度为1，表示Z方向

    def forward(self, x):
        x = x.unsqueeze(1)

        # 多尺度卷积层
        conv1_out = self.conv1(x)
        conv3_out = self.conv3(x)
        conv5_out = self.conv5(x)

        # 全连接层映射
        conv1_out = self.fc1(conv1_out)
        conv3_out = self.fc3(conv3_out)
        conv5_out = self.fc5(conv5_out)

        # 添加位置编码
        conv1_out = self.pos_embedding(conv1_out) + conv1_out
        conv3_out = self.pos_embedding(conv3_out) + conv3_out
        conv5_out = self.pos_embedding(conv5_out) + conv5_out

        # 多头注意力处理
        attn_output3, _ = self.multihead_attn1(conv1_out, conv3_out, conv3_out)
        attn_output5, _ = self.multihead_attn2(conv1_out, conv5_out, conv5_out)

        # Transformer编码
        out1 = self.transformer_encoder_1(conv1_out)
        out3 = self.transformer_encoder_3(attn_output3)
        out5 = self.transformer_encoder_5(attn_output5)

        # 特征融合与解耦
        combined = torch.cat((out1, out3, out5), dim=2)
        combined = self.norm(combined)
        combined = self.fc_out(combined)
        combined = combined.mean(dim=1)

        # 输出Z方向载荷值
        return self.final_fc(combined)

