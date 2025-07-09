import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from utils.text_parser import load_events_for_dataset
import warnings

warnings.filterwarnings('ignore')

class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        
        # 提取数据集名称用于加载事件信息
        self.dataset_name = os.path.splitext(data_path)[0]
        
        # 设置事件文本文件后缀，默认为 _combined_info.txt
        self.event_text_suffix = getattr(self.args, 'event_text_suffix', '_combined_info.txt')
        
        # 确定事件文本信息的根目录
        self.event_root_path = root_path
        
        # 如果数据文件不在目录中，则尝试使用数据集名称作为目录
        if not any(f.endswith(self.event_text_suffix) for f in os.listdir(self.event_root_path)):
            possible_dir = os.path.join(os.path.dirname(root_path), self.dataset_name)
            if os.path.exists(possible_dir):
                self.event_root_path = possible_dir
        
        # 查找匹配的事件文本文件
        self.event_file = None
        for f in os.listdir(self.event_root_path):
            if f.endswith(self.event_text_suffix):
                self.event_file = os.path.join(self.event_root_path, f)
                break
                    
        print(f"数据文件夹: {self.dataset_name}")
        print(f"事件文本目录: {self.event_root_path}")
        print(f"事件文件后缀: {self.event_text_suffix}")
        print(f"找到事件文件: {self.event_file if self.event_file else '未找到'}")
        
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        
        # 使用从命令行传入的比例参数
        train_ratio = self.args.train_ratio
        val_ratio = self.args.val_ratio
        
        # 确保比例合理
        if train_ratio + val_ratio >= 1.0:
            raise ValueError(f'训练集比例({train_ratio:.2f}) + 验证集比例({val_ratio:.2f}) 必须小于1')
            
        num_train = int(len(df_raw) * train_ratio)
        num_val = int(len(df_raw) * val_ratio)
        num_test = len(df_raw) - num_train - num_val
        
        print(f'数据集划分: 训练集={num_train}({train_ratio*100:.1f}%), 验证集={num_val}({val_ratio*100:.1f}%), 测试集={num_test}({(1-train_ratio-val_ratio)*100:.1f}%)')
        
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        # 使用dayfirst=True参数来处理日/月/年格式
        df_stamp['date'] = pd.to_datetime(df_stamp.date, dayfirst=True)
        # 保存日期信息供后续使用
        self.df_dates = df_stamp['date'].values
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        # 加载事件文本信息
        self.events_map = {}
        self.date_to_event = {}
        
        if self.event_file and os.path.exists(self.event_file):
            try:
                print(f"加载事件文件: {self.event_file}")
                self.events_map = load_events_for_dataset(self.dataset_name, os.path.dirname(self.event_file), 
                                                         os.path.basename(self.event_file))
                
                # 创建日期到事件文本的映射
                for date_str, event_text in self.events_map.items():
                    self.date_to_event[date_str] = event_text
                    
                print(f"成功加载事件信息，共 {len(self.events_map)} 个事件")
            except Exception as e:
                print(f"加载事件信息失败: {e}")
                self.events_map = {}
                self.date_to_event = {}
        
        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        # 获取预测日期对应的事件文本
        event_text = ""
        if hasattr(self, 'date_to_event') and len(self.date_to_event) > 0:
            # 尝试获取预测日期
            try:
                # 获取预测窗口的第一个日期
                pred_date_idx = r_begin + self.label_len
                if pred_date_idx < len(self.df_dates):
                    # 格式化日期为字符串
                    pred_date = pd.Timestamp(self.df_dates[pred_date_idx]).strftime('%Y-%m-%d')
                    
                    # 查找对应日期的事件文本
                    event_text = self.date_to_event.get(pred_date, "")
                    
                    # 如果有事件文本，输出调试信息
                    if event_text:
                        print(f"找到日期 {pred_date} 的事件文本: {event_text[:50]}...")
            except Exception as e:
                print(f"获取预测日期失败: {e}")
                event_text = ""
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark, event_text

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)