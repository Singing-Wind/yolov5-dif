import torch
import torch.nn as nn

class TransformerMLPHead(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, output_dim=4, num_layers=4, num_heads=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim

        # 将输入 [b, C] 扩展为 [b, 1, C]，作为 transformer token
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim, 
            dropout=dropout, 
            batch_first=True  # 注意启用 batch_first 以便支持 [b, 1, C]
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 最终映射至输出维度
        self.head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        # x: [b, C] -> [b, 1, C]
        x = x.unsqueeze(1)

        # transformer output: [b, 1, C]
        x = self.transformer(x)

        # 取出 token 向量（[b, 1, C] -> [b, C]）
        x = x.squeeze(1)

        # 输出为 [b, 4]
        out = self.head(x)
        return out