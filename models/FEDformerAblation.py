import torch
import torch.nn as nn
import torch.nn.functional as F
from models.FEDformerWithText import FEDformerWithText

class FEDformerAblation(FEDformerWithText):
    def __init__(self, configs, ablation_type='full', **kwargs):
        """
        FEDformer with ablation study support.
        
        Args:
            configs: Configuration object
            ablation_type: Type of ablation study
                - 'no_text': No text input
                - 'frozen_bert': Use frozen BERT
                - 'additive': Use additive fusion
                - 'late_fusion': Fuse at encoder output
                - 'full': Original implementation
        """
        self.ablation_type = ablation_type
        
        # For 'no_text' ablation, disable text input
        if ablation_type == 'no_text':
            configs.use_event_text = 0
        
        super().__init__(configs, **kwargs)
        
        # For 'frozen_bert' ablation, freeze BERT parameters
        if ablation_type == 'frozen_bert' and hasattr(self, 'text_embedding') and hasattr(self.text_embedding, 'bert'):
            for param in self.text_embedding.bert.parameters():
                param.requires_grad = False
    
    def _fuse_text_embedding(self, x_enc, event_texts):
        """Modified fusion for ablation studies"""
        if not self.use_text or event_texts is None:
            return x_enc
            
        # Ensure components are initialized
        if not self.initialized:
            self._initialize_components(x_enc.device)
        
        # Get text embeddings
        text_emb = self.text_embedding(event_texts)  # [batch_size, d_model]
        
        # Project and normalize
        text_emb = self.text_projection(text_emb)
        text_emb = self.text_norm(text_emb)
        
        # For additive fusion
        if self.ablation_type == 'additive':
            seq_len = x_enc.size(1)
            text_emb = text_emb.unsqueeze(1).expand(-1, seq_len, -1)
            return x_enc + text_emb  # Simple addition
            
        # For late fusion, just return the original features
        # The fusion will be done in the forward method
        if self.ablation_type == 'late_fusion':
            return x_enc, text_emb
            
        # Original gated fusion
        seq_len = x_enc.size(1)
        text_emb = text_emb.unsqueeze(1).expand(-1, seq_len, -1)
        gate_input = torch.cat([x_enc, text_emb], dim=-1)
        gate = self.gate(gate_input)
        return gate * x_enc + (1 - gate) * text_emb
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, event_texts=None, mask=None):
        # For 'no_text' ablation, ignore text input
        if self.ablation_type == 'no_text':
            event_texts = None
        
        # For late fusion, handle text at the end
        if self.ablation_type == 'late_fusion' and event_texts is not None:
            # Get encoder output
            enc_out = self.enc_embedding(x_enc, x_mark_enc)
            enc_out, attns = self.encoder(enc_out, attn_mask=None)
            
            # Get text features
            if not self.initialized:
                self._initialize_components(x_enc.device)
            text_emb = self.text_embedding(event_texts)
            text_emb = self.text_projection(text_emb)
            text_emb = self.text_norm(text_emb)
            text_emb = text_emb.unsqueeze(1).expand(-1, enc_out.size(1), -1)
            
            # Fuse at encoder output
            gate_input = torch.cat([enc_out, text_emb], dim=-1)
            gate = self.gate(gate_input)
            enc_out = gate * enc_out + (1 - gate) * text_emb
        else:
            # Original forward with modified _fuse_text_embedding
            return super().forward(x_enc, x_mark_enc, x_dec, x_mark_dec, event_texts, mask)
            
        # Rest of the forward pass for late fusion
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
            seasonal_init, trend_init = self.decomp(x_enc)
            trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
            seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
            
            dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
            seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init)
            
            dec_out = trend_part + seasonal_part
            return dec_out[:, -self.pred_len:, :]
            
        elif self.task_name == 'imputation':
            dec_out = self.projection(enc_out)
            return dec_out
            
        elif self.task_name == 'anomaly_detection':
            dec_out = self.projection(enc_out)
            return dec_out
