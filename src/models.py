import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len=1):
        super(PositionEmbedding, self).__init__()
        self.pe = torch.zeros(max_len, d_model, device='cuda' if torch.cuda.is_available() else 'cpu')
        position = torch.arange(0, max_len, dtype=torch.float, device=self.pe.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float, device=self.pe.device) * (
                -torch.log(torch.tensor(10000.0, device=self.pe.device)) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)  # 增加 batch 维度

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]  # 仅使用输入序列长度范围内的位置编码


class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, num_hidden_layers, d_model, nhead):
        super(TransformerModel, self).__init__()

        self.input_layer = nn.Linear(input_size, d_model)
        self.pos_embedding = PositionEmbedding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=0.2
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_hidden_layers)

        self.fc_x = nn.Linear(d_model, output_size)  # 输出层
        self.fc_y = nn.Linear(d_model, output_size)
        self.fc_z = nn.Linear(d_model, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.input_layer(x)
        position = self.pos_embedding(x)
        x = x + position

        x = self.encoder(x)

        x_x = self.fc_x(x[:, -1, :])
        x_y = self.fc_y(x[:, -1, :])
        x_z = self.fc_z(x[:, -1, :])
        out = torch.cat([x_x, x_y, x_z], dim=1)  # 沿轴 1 连接预测
        return out


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_hidden_layers, batch_first=True, bidirectional=True)

        self.fc_x = nn.Linear(hidden_size * 2, output_size)  # Output head for X-axis
        self.fc_y = nn.Linear(hidden_size * 2, output_size)  # Output head for Y-axis
        self.fc_z = nn.Linear(hidden_size * 2, output_size)  # Output head for Z-axis

    def forward(self, x):
        x = x.unsqueeze(1)
        output, _ = self.lstm(x)

        x_x = self.fc_x(output[:, -1, :])
        x_y = self.fc_y(output[:, -1, :])
        x_z = self.fc_z(output[:, -1, :])

        out = torch.stack([x_x, x_y, x_z], dim=0)
        out = torch.squeeze(out, dim=2)
        out = out.t()
        return out


class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=0)
        self.linear = nn.Linear(7, 3)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(9, 64)
        self.fc2 = nn.Linear(64, 64)

        self.fc_x = nn.Linear(64, 1)  # Output head for X-axis
        self.fc_y = nn.Linear(64, 1)  # Output head for Y-axis
        self.fc_z = nn.Linear(64, 1)  # Output head for Z-axis

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        x_x = self.fc_x(x)
        x_y = self.fc_y(x)
        x_z = self.fc_z(x)

        out = torch.stack([x_x, x_y, x_z], dim=0)
        out = torch.squeeze(out, dim=2)
        out = out.t()
        return out


class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3),
            nn.ReLU(),

            # nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2),
            # nn.ReLU()
        )

        self.lstm = nn.LSTM(input_size=8, hidden_size=64, num_layers=2)

        self.dropout = nn.Dropout(0.2)

        self.fc_x = nn.Linear(64, 1)  # Output head for X-axis
        self.fc_y = nn.Linear(64, 1)  # Output head for Y-axis
        self.fc_z = nn.Linear(64, 1)  # Output head for Z-axis

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.permute(1, 0, 2)
        x, _ = self.lstm(x)
        x = self.dropout(x)  # Apply dropout

        x_x = self.fc_x(x[-1, :, :])
        x_y = self.fc_y(x[-1, :, :])
        x_z = self.fc_z(x[-1, :, :])

        out = torch.stack([x_x, x_y, x_z], dim=0)
        out = torch.squeeze(out, dim=2)
        out = out.t()
        return out


class Bottleneck(torch.nn.Module):
    def __init__(self, In_channel, Med_channel, Out_channel, downsample=False):
        super(Bottleneck, self).__init__()
        self.stride = 1
        if downsample:
            self.stride = 2

        self.layer = torch.nn.Sequential(
            torch.nn.Conv1d(In_channel, Med_channel, 1, self.stride),
            torch.nn.BatchNorm1d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv1d(Med_channel, Med_channel, 3, padding=1),
            torch.nn.BatchNorm1d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv1d(Med_channel, Out_channel, 1),
            torch.nn.BatchNorm1d(Out_channel),
            torch.nn.ReLU(),
        )

        if In_channel != Out_channel or downsample:
            self.res_layer = torch.nn.Conv1d(In_channel, Out_channel, 1, self.stride)
        else:
            self.res_layer = None

    def forward(self, x):
        residual = self.res_layer(x) if self.res_layer is not None else x
        return self.layer(x) + residual


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_size=3):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 根据输出长度计算第二个卷积层的参数
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 如果输入通道和输出通道不一致，需要添加1x1卷积来调整
        self.adjust = nn.Conv1d(in_channels, out_channels, kernel_size=1,
                                stride=stride) if in_channels != out_channels else None

        # 添加一个自适应平均池化层，输出固定大小
        self.pool = nn.AdaptiveAvgPool1d(output_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        identity = x  # 保存输入以用于残差连接

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 如果需要调整输入维度，应用1x1卷积
        if self.adjust:
            identity = self.adjust(identity)

        out += identity  # 残差连接
        out = self.relu(out)  # 激活函数

        out = self.pool(out)  # 自适应平均池化层
        out = out.view(out.size(0), -1)

        return out
