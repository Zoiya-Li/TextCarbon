import torch
import torch.nn as nn
from einops import rearrange, repeat
from models.Crossformer import Model as CrossformerModel
from models.TextEmbedding import SimpleTextEmbedding

class CrossformerWithText(CrossformerModel):
    def __init__(self, configs):
        super(CrossformerWithText, self).__init__(configs)
        
        self.d_model = configs.d_model
        self.use_text = hasattr(configs, 'use_event_text') and configs.use_event_text == 1
        print(f"是否使用事件文本嵌入: {self.use_text}")
        
        # 初始化文本嵌入模块（延迟加载）
        self.text_embedding = SimpleTextEmbedding(self.d_model)
        
        # 文本投影层
        self.text_projection = None
        self.text_norm = None
        self.gate = None
        
        # 标记是否已初始化
        self.initialized = False
        
    def _initialize_components(self, device):
        if self.initialized:
            return
            
        # 初始化文本相关组件
        if self.use_text:
            self.text_embedding._load_model(device)
            self.text_projection = nn.Linear(self.d_model, self.d_model).to(device)
            self.text_norm = nn.LayerNorm(self.d_model).to(device)
            self.gate = nn.Sequential(
                nn.Linear(2 * self.d_model, self.d_model),
                nn.Sigmoid()
            ).to(device)
            
        self.initialized = True

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, event_texts=None):
        # 确保组件已初始化
        if not self.initialized:
            self._initialize_components(x_enc.device)
            
        # 原始Crossformer处理
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            # 获取时间序列的嵌入
            x_enc_embed, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
            x_enc_embed = rearrange(x_enc_embed, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)
            
            # 添加位置编码
            x_enc_embed = x_enc_embed.to(x_enc.device) + self.enc_pos_embedding.to(x_enc.device)
            x_enc_embed = self.pre_norm(x_enc_embed)
            
            # 处理文本信息
            if self.use_text and event_texts is not None and len(event_texts) > 0:
                # 确保文本组件已初始化
                if not self.initialized:
                    self._initialize_components(x_enc.device)
                    
                # 获取文本嵌入
                text_embed = self.text_embedding(event_texts)
                text_embed = self.text_projection(text_embed)
                text_embed = self.text_norm(text_embed)
                
                # 扩展文本嵌入以匹配时间序列的维度
                text_embed = text_embed.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, D]
                text_embed = text_embed.expand(-1, x_enc_embed.size(1), x_enc_embed.size(2), -1)
                
                # 门控融合
                gate_input = torch.cat([x_enc_embed, text_embed], dim=-1)
                gate = self.gate(gate_input)
                x_enc_embed = gate * x_enc_embed + (1 - gate) * text_embed
                x_enc_embed = self.pre_norm(x_enc_embed)
            
            # 通过编码器
            enc_out, _ = self.encoder(x_enc_embed)
            
            # 解码器处理
            dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat=x_enc_embed.shape[0])
            dec_out = self.decoder(dec_in, enc_out)
            
            return dec_out[:, -self.pred_len:, :]
            
        # 其他任务类型保持不变
        elif self.task_name == 'imputation':
            # 处理插值任务
            x_enc_embed, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
            x_enc_embed = rearrange(x_enc_embed, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)
            
            # 添加位置编码
            x_enc_embed = x_enc_embed.to(x_enc.device) + self.enc_pos_embedding.to(x_enc.device)
            x_enc_embed = self.pre_norm(x_enc_embed)
            
            # 通过编码器
            enc_out = self.encoder(x_enc_embed)
            
            # 输出层
            output = self.projection(enc_out)
            output = output.permute(0, 2, 1, 3)[:, :, :, 0]  # [B, L, D]
            
            return output
            
        elif self.task_name == 'anomaly_detection':
            # 处理异常检测任务
            x_enc_embed, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
            x_enc_embed = rearrange(x_enc_embed, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)
            
            # 添加位置编码
            x_enc_embed = x_enc_embed.to(x_enc.device) + self.enc_pos_embedding.to(x_enc.device)
            x_enc_embed = self.pre_norm(x_enc_embed)
            
            # 通过编码器
            enc_out = self.encoder(x_enc_embed)
            
            # 输出层
            output = self.projection(enc_out)
            output = output.permute(0, 2, 1, 3)[:, :, :, 0]  # [B, L, D]
            
            return output
            
        elif self.task_name == 'classification':
            # 处理分类任务
            x_enc_embed, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
            x_enc_embed = rearrange(x_enc_embed, '(b d) seg_num d_model -> b d seg_num d_model', d=n_vars)
            
            # 添加位置编码
            x_enc_embed = x_enc_embed.to(x_enc.device) + self.enc_pos_embedding.to(x_enc.device)
            x_enc_embed = self.pre_norm(x_enc_embed)
            
            # 通过编码器
            enc_out = self.encoder(x_enc_embed)
            
            # 输出层
            output = self.act(enc_out)  # 激活函数
            output = output.reshape(output.shape[0], -1)  # 展平
            output = self.dropout(output)
            output = self.projection(output)  # 分类层
            
            return output
            
        else:
            raise ValueError(f"Unknown task: {self.task_name}")
