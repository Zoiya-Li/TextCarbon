import torch
import torch.nn as nn
import torch.nn.functional as F
from models.FEDformer import Model as FEDformerModel
from models.TextEmbedding import SimpleTextEmbedding

class FEDformerWithText(FEDformerModel):
    def __init__(self, configs, version='fourier', mode_select='random', modes=32):
        """
        FEDformer with text embedding support.
        
        Args:
            configs: Configuration object containing model parameters
            version: str, 'fourier' or 'wavelet'
            mode_select: str, 'random' or 'low'
            modes: int, number of modes to select
        """
        super(FEDformerWithText, self).__init__(configs, version, mode_select, modes)
        
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
        """Initialize text-related components"""
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
    
    def _fuse_text_embedding(self, x_enc, event_texts):
        """
        Fuse text embedding with time series features
        
        Args:
            x_enc: [batch_size, seq_len, d_model], time series features
            event_texts: list of str, event texts for each sample in batch
            
        Returns:
            fused_features: [batch_size, seq_len, d_model], fused features
        """
        if not self.use_text or event_texts is None:
            return x_enc
            
        # 确保组件已初始化
        if not self.initialized:
            self._initialize_components(x_enc.device)
        
        # 获取文本嵌入 [batch_size, d_model]
        text_emb = self.text_embedding(event_texts)  # [batch_size, d_model]
        
        # 投影和归一化
        text_emb = self.text_projection(text_emb)  # [batch_size, d_model]
        text_emb = self.text_norm(text_emb)  # [batch_size, d_model]
        
        # 扩展文本嵌入以匹配时间序列长度 [batch_size, seq_len, d_model]
        seq_len = x_enc.size(1)
        text_emb = text_emb.unsqueeze(1).expand(-1, seq_len, -1)
        
        # 使用门控机制融合特征
        gate_input = torch.cat([x_enc, text_emb], dim=-1)  # [batch_size, seq_len, 2*d_model]
        gate = self.gate(gate_input)  # [batch_size, seq_len, d_model]
        
        # 使用门控机制融合特征
        fused_features = gate * x_enc + (1 - gate) * text_emb
        
        return fused_features
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, event_texts=None, mask=None):
        """
        Forward pass with text embedding support
        
        Args:
            x_enc: [batch_size, seq_len, enc_in]
            x_mark_enc: [batch_size, seq_len, mark_dim]
            x_dec: [batch_size, label_len + pred_len, dec_in]
            x_mark_dec: [batch_size, label_len + pred_len, mark_dim]
            event_texts: list of str, event texts for each sample in batch
            mask: mask for imputation task
            
        Returns:
            output: model output
        """
        # 确保组件已初始化
        if not self.initialized:
            self._initialize_components(x_enc.device)
        
        # 原始FEDformer处理
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # 获取时间序列的嵌入
            enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [batch_size, seq_len, d_model]
            
            # 如果使用文本嵌入，则融合文本特征
            if self.use_text and event_texts is not None:
                enc_out = self._fuse_text_embedding(enc_out, event_texts)
            
            # 编码器处理
            enc_out, attns = self.encoder(enc_out, attn_mask=None)
            
            # 解码器输入准备
            mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
            seasonal_init, trend_init = self.decomp(x_enc)
            trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
            seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
            
            # 解码器
            dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
            seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init)
            
            # 最终输出
            dec_out = trend_part + seasonal_part
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
            
        elif self.task_name == 'imputation':
            # 编码器处理
            enc_out = self.enc_embedding(x_enc, x_mark_enc)
            
            # 如果使用文本嵌入，则融合文本特征
            if self.use_text and event_texts is not None:
                enc_out = self._fuse_text_embedding(enc_out, event_texts)
            
            enc_out, attns = self.encoder(enc_out, attn_mask=None)
            dec_out = self.projection(enc_out)
            return dec_out  # [B, L, D]
            
        elif self.task_name == 'anomaly_detection':
            # 编码器处理
            enc_out = self.enc_embedding(x_enc, None)
            
            # 如果使用文本嵌入，则融合文本特征
            if self.use_text and event_texts is not None:
                enc_out = self._fuse_text_embedding(enc_out, event_texts)
            
            enc_out, attns = self.encoder(enc_out, attn_mask=None)
            dec_out = self.projection(enc_out)
            return dec_out  # [B, L, D]
            
        elif self.task_name == 'classification':
            # 编码器处理
            enc_out = self.enc_embedding(x_enc, x_mark_enc)
            
            # 如果使用文本嵌入，则融合文本特征
            if self.use_text and event_texts is not None:
                enc_out = self._fuse_text_embedding(enc_out, event_texts)
            
            enc_out, attns = self.encoder(enc_out, attn_mask=None)
            
            # 分类头
            output = F.gelu(enc_out)
            output = self.dropout(output)
            output = output * x_mark_enc.unsqueeze(-1)
            output = output.reshape(output.shape[0], -1)
            output = self.projection(output)
            return output  # [B, num_class]
            
        return None
