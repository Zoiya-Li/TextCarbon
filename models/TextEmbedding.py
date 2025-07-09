import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class BertTextEmbedding(nn.Module):
    """
    使用预训练的中文BERT模型进行文本嵌入
    """
    def __init__(self, d_model, max_length=128, cache_dir=None):
        super(BertTextEmbedding, self).__init__()
        self.d_model = d_model
        self.max_length = max_length
        self.model_name = 'bert-base-chinese'
        
        # 设置缓存目录
        self.cache_dir = cache_dir
        
        # 初始化模型组件
        self.bert = None
        self.tokenizer = None
        self.projection = None
        self.initialized = False
        self.dummy_param = nn.Parameter(torch.zeros(1))
        
    def _load_model(self, device=None):
        """加载BERT模型和tokenizer"""
        if self.initialized:
            return
            
        if device is None:
            device = self.dummy_param.device
        
        # 将设备转换为torch.device对象（如果是字符串）
        if isinstance(device, str):
            device = torch.device(device)
            
        try:
            # 使用系统默认设置加载tokenizer和模型
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.bert = BertModel.from_pretrained(self.model_name)
            
            # 将模型移动到指定设备
            self.bert = self.bert.to(device)
            
            # 冻结BERT参数，加快训练速度
            for param in self.bert.parameters():
                param.requires_grad = False
                
            # 初始化投影层
            self.projection = nn.Linear(self.bert.config.hidden_size, self.d_model).to(device)
            self.dropout = nn.Dropout(0.1).to(device)
            
            self.initialized = True
            
        except Exception as e:
            error_msg = f"Failed to load BERT model: {str(e)}\n"
            error_msg += "\n" + "="*80 + "\n"
            error_msg += "Failed to load BERT model. Please check the following:\n"
            error_msg += "1. Make sure you have internet connection to access Hugging Face Hub\n"
            error_msg += "2. Try downloading the model manually:\n"
            error_msg += "   python -c \"from transformers import BertModel; BertModel.from_pretrained('bert-base-chinese')\"\n"
            error_msg += "3. If you're behind a proxy, set the proxy environment variables:\n"
            error_msg += "   export HTTP_PROXY=http://your-proxy:port\n"
            error_msg += "   export HTTPS_PROXY=http://your-proxy:port\n"
            error_msg += "="*80
            print(error_msg)
            raise
        
    def forward(self, text_list):
        """
        将文本列表转换为嵌入向量
        
        Args:
            text_list: 文本列表，每个元素是一个字符串
            
        Returns:
            text_embedding: 文本嵌入向量 [batch_size, d_model]
        """
        try:
            # 延迟初始化模型，确保在正确的设备上
            if not self.initialized:
                self._load_model(self.dummy_param.device)
                
            batch_size = len(text_list)
            device = self.dummy_param.device
            
            # 如果文本列表为空或全部为空字符串，返回零向量
            if batch_size == 0 or all(not text for text in text_list):
                return torch.zeros((batch_size, self.d_model), device=device)
            
            # 处理非空文本
            valid_texts = [text for text in text_list if text]
            if not valid_texts:
                return torch.zeros((batch_size, self.d_model), device=device)
        
            # 对文本进行分词和编码
            encoded_input = self.tokenizer(
                valid_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt',
                return_attention_mask=True
            )
            
            # 确保输入张量在正确的设备上
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
            
            # 获取BERT的输出
            with torch.no_grad():
                outputs = self.bert(**encoded_input)
                
            # 使用[CLS]标记的隐藏状态作为整个句子的表示
            # [batch_size, hidden_size]
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            
            # 投影到目标维度
            projected_embed = self.projection(cls_embedding)
            projected_embed = self.dropout(projected_embed)
            
            # 对于空文本，使用零向量
            final_embed = torch.zeros((batch_size, self.d_model), device=device)
            
            # 将有效文本的嵌入放入正确的位置
            valid_idx = 0
            for i in range(batch_size):
                if text_list[i]:  # 如果不是空字符串
                    final_embed[i] = projected_embed[valid_idx]
                    valid_idx += 1
                    
            return final_embed
                    
        except Exception as e:
            # 返回与预期形状相同的零张量
            return torch.zeros((len(text_list), self.d_model), device=self.dummy_param.device)

# 为了兼容性，保留原来的类名
SimpleTextEmbedding = BertTextEmbedding
