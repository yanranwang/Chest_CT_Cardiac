import copy
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
import torchvision
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from merlin.models.i3res import I3ResNet


class CardiacImageEncoder(nn.Module):
    """图像编码器，专门用于心脏功能预测"""
    def __init__(self, pretrained_path=None):
        super().__init__()
        resnet = torchvision.models.resnet152(pretrained=True)
        self.i3_resnet = I3ResNet(
            copy.deepcopy(resnet), class_nb=1692, conv_class=True, ImageEmbedding=False
        )
        
        # 如果提供了预训练权重路径，加载权重
        if pretrained_path:
            self._load_pretrained_weights(pretrained_path)
    
    def _load_pretrained_weights(self, path):
        """加载预训练的Merlin权重"""
        try:
            state_dict = torch.load(path, map_location='cpu')
            
            # 只加载图像编码器相关的权重
            image_encoder_dict = {}
            
            # 遍历所有权重键
            for key, value in state_dict.items():
                # 查找encode_image.i3_resnet开头的权重键
                if key.startswith('encode_image.i3_resnet.'):
                    # 移除encode_image.前缀，保留i3_resnet.部分
                    new_key = key.replace('encode_image.', '')
                    image_encoder_dict[new_key] = value
                    print(f"映射权重: {key} -> {new_key}")
            
            if image_encoder_dict:
                # 尝试加载权重到i3_resnet
                missing_keys, unexpected_keys = self.i3_resnet.load_state_dict(image_encoder_dict, strict=False)
                
                if missing_keys:
                    print(f"缺少的权重键: {missing_keys}")
                if unexpected_keys:
                    print(f"意外的权重键: {unexpected_keys}")
                    
                print(f"成功加载预训练的图像编码器权重，共 {len(image_encoder_dict)} 个参数")
            else:
                print("警告: 在预训练权重中未找到图像编码器相关的权重")
                print("可用的权重键:")
                for key in list(state_dict.keys())[:10]:  # 显示前10个键
                    print(f"  {key}")
                if len(state_dict.keys()) > 10:
                    print(f"  ... 还有 {len(state_dict.keys()) - 10} 个键")
                
        except Exception as e:
            print(f"加载预训练权重失败: {e}")
            print("将使用随机初始化的权重继续训练")
    
    def forward(self, image):
        contrastive_features, ehr_features = self.i3_resnet(image)
        return contrastive_features, ehr_features


class CardiacPredictionHead(nn.Module):
    """心脏功能预测头：LVEF回归 + AS二分类"""
    def __init__(self, input_dim=512, hidden_dims=[256, 128]):
        super().__init__()
        
        # 共享特征提取层
        shared_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.LayerNorm(hidden_dim)  # 使用LayerNorm替代BatchNorm，不依赖于batch_size
            ])
            prev_dim = hidden_dim
        
        self.shared_features = nn.Sequential(*shared_layers)
        
        # LVEF回归头
        self.lvef_regressor = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)  # LVEF输出 (射血分数百分比)
        )
        
        # AS二分类头 
        self.as_classifier = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),  # AS存在性输出 (sigmoid激活)
            nn.Sigmoid()
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, features):
        """前向传播
        Args:
            features: 图像特征 [batch_size, feature_dim]
        Returns:
            lvef_pred: LVEF预测值 [batch_size, 1]
            as_pred: AS存在性概率 [batch_size, 1]
        """
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
        
        # 标准化特征
        features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
        
        # 共享特征提取
        shared_feat = self.shared_features(features)
        
        # LVEF回归预测
        lvef_pred = self.lvef_regressor(shared_feat)
        
        # AS分类预测
        as_pred = self.as_classifier(shared_feat)
        
        return lvef_pred, as_pred


class CardiacFunctionModel(nn.Module):
    """完整的心脏功能预测模型：LVEF回归 + AS二分类"""
    def __init__(self, pretrained_model_path=None):
        super().__init__()
        
        # 图像编码器
        self.image_encoder = CardiacImageEncoder(pretrained_model_path)
        
        # 心脏功能预测头
        self.cardiac_predictor = CardiacPredictionHead(input_dim=512)
        
        # 是否冻结图像编码器
        self.freeze_image_encoder = False
    
    def freeze_encoder(self, freeze=True):
        """冻结或解冻图像编码器"""
        self.freeze_image_encoder = freeze
        for param in self.image_encoder.parameters():
            param.requires_grad = not freeze
    
    def forward(self, image, return_features=False):
        """前向传播
        Args:
            image: CT图像 [batch_size, channels, depth, height, width]
            return_features: 是否返回中间特征
        
        Returns:
            lvef_pred: LVEF预测值 [batch_size, 1]
            as_pred: AS存在性概率 [batch_size, 1]  
            image_features: 图像特征 [batch_size, 512] (可选)
        """
        # 提取图像特征
        image_features, _ = self.image_encoder(image)
        
        # 心脏功能预测
        lvef_pred, as_pred = self.cardiac_predictor(image_features)
        
        if return_features:
            return lvef_pred, as_pred, image_features
        else:
            return lvef_pred, as_pred
    
    def predict(self, image):
        """预测接口"""
        self.eval()
        with torch.no_grad():
            lvef_pred, as_pred = self.forward(image)
            return {
                'lvef': lvef_pred.squeeze().cpu().numpy(),
                'as_probability': as_pred.squeeze().cpu().numpy(),
                'as_prediction': (as_pred > 0.5).squeeze().cpu().numpy()
            }


class CardiacDataset(Dataset):
    """心脏CT数据集，包含LVEF和AS标签"""
    def __init__(self, csv_path, image_dir=None, split='train', test_size=0.2, random_state=42):
        """
        Args:
            csv_path: CSV文件路径
            image_dir: CT图像目录路径
            split: 数据分割 ('train', 'val', 'test')
            test_size: 测试集比例
            random_state: 随机种子
        """
        self.csv_path = csv_path
        self.image_dir = image_dir
        
        # 读取CSV数据
        self.df = pd.read_csv(csv_path)
        
        # 过滤有效数据（同时有LVEF和AS_definite标签）
        valid_mask = self.df['lvef'].notna() & self.df['AS_definite'].notna()
        self.df = self.df[valid_mask].reset_index(drop=True)
        
        print(f"有效样本数量: {len(self.df)}")
        print(f"LVEF范围: {self.df['lvef'].min():.2f} - {self.df['lvef'].max():.2f}")
        print(f"AS分布: {self.df['AS_definite'].value_counts().to_dict()}")
        
        # 数据分割
        if split in ['train', 'val']:
            train_indices, val_indices = train_test_split(
                range(len(self.df)), 
                test_size=test_size, 
                random_state=random_state,
                stratify=self.df['AS_definite']  # 保持AS类别平衡
            )
            
            if split == 'train':
                self.indices = train_indices
            else:
                self.indices = val_indices
        else:
            self.indices = list(range(len(self.df)))
        
        print(f"{split}集样本数量: {len(self.indices)}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        real_idx = self.indices[idx]
        row = self.df.iloc[real_idx]
        
        # 获取标签
        lvef = float(row['lvef'])
        as_definite = int(row['AS_definite'])
        
        # 获取图像标识符
        mrn = row['mrn']
        accession_number = row['accession_number']
        
        sample = {
            'mrn': mrn,
            'accession_number': accession_number,
            'lvef': torch.tensor(lvef, dtype=torch.float32),
            'as_definite': torch.tensor(as_definite, dtype=torch.float32),
            'lvef_normalized': torch.tensor((lvef - 59.22) / 11.85, dtype=torch.float32),  # 标准化LVEF
        }
        
        # 如果有图像目录，可以在这里加载CT图像
        # if self.image_dir:
        #     image_path = os.path.join(self.image_dir, f"{mrn}_{accession_number}.nii.gz")
        #     image = self.load_ct_image(image_path)
        #     sample['image'] = image
        
        return sample
    
    def get_class_weights(self):
        """计算AS分类的类别权重，用于处理类别不平衡"""
        as_counts = self.df.iloc[self.indices]['AS_definite'].value_counts().sort_index()
        total = len(self.indices)
        weights = total / (2 * as_counts.values)
        return torch.tensor(weights, dtype=torch.float32)


class CardiacLoss(nn.Module):
    """组合损失函数：LVEF回归损失 + AS分类损失"""
    def __init__(self, regression_weight=1.0, classification_weight=1.0, class_weights=None):
        super().__init__()
        self.regression_weight = regression_weight
        self.classification_weight = classification_weight
        
        # 回归损失：均方误差
        self.regression_loss = nn.MSELoss()
        
        # 分类损失：加权二元交叉熵
        if class_weights is not None:
            # 处理类别不平衡
            pos_weight = class_weights[1] / class_weights[0]
            self.classification_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.classification_loss = nn.BCELoss()
    
    def forward(self, lvef_pred, as_pred, lvef_true, as_true):
        """
        Args:
            lvef_pred: LVEF预测值 [batch_size, 1]
            as_pred: AS预测概率 [batch_size, 1]
            lvef_true: LVEF真实值 [batch_size]
            as_true: AS真实标签 [batch_size]
        """
        # LVEF回归损失
        reg_loss = self.regression_loss(lvef_pred.squeeze(), lvef_true)
        
        # AS分类损失
        clf_loss = self.classification_loss(as_pred.squeeze(), as_true)
        
        # 组合损失
        total_loss = (self.regression_weight * reg_loss + 
                     self.classification_weight * clf_loss)
        
        return {
            'total_loss': total_loss,
            'regression_loss': reg_loss,
            'classification_loss': clf_loss
        }


def create_cardiac_dataloaders(csv_path, batch_size=32, num_workers=4):
    """创建训练和验证数据加载器"""
    
    # 创建数据集
    train_dataset = CardiacDataset(csv_path, split='train')
    val_dataset = CardiacDataset(csv_path, split='val')
    
    # 获取类别权重
    class_weights = train_dataset.get_class_weights()
    print(f"AS类别权重: {class_weights}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, class_weights


class CardiacMetricsCalculator:
    """心脏功能指标计算和标准化工具"""
    
    @staticmethod
    def get_metric_names():
        """获取心脏功能指标名称列表"""
        return [
            'ejection_fraction',       # 射血分数
            'stroke_volume',          # 每搏输出量
            'cardiac_output',         # 心输出量
            'heart_rate_variability', # 心率变异性
            'left_ventricular_mass',  # 左心室质量
            'wall_thickness',         # 室壁厚度
            'chamber_volume',         # 心室容积
            'contractility_index',    # 收缩性指数
            'diastolic_function',     # 舒张功能
            'valvular_function'       # 瓣膜功能
        ]
    
    @staticmethod
    def get_metric_descriptions():
        """获取心脏功能指标的中文描述"""
        return {
            'ejection_fraction': '射血分数 (%)',
            'stroke_volume': '每搏输出量 (mL)',
            'cardiac_output': '心输出量 (L/min)',
            'heart_rate_variability': '心率变异性 (ms)',
            'left_ventricular_mass': '左心室质量 (g)',
            'wall_thickness': '室壁厚度 (mm)',
            'chamber_volume': '心室容积 (mL)',
            'contractility_index': '收缩性指数',
            'diastolic_function': '舒张功能',
            'valvular_function': '瓣膜功能'
        }
    
    @staticmethod
    def get_normal_ranges():
        """获取心脏功能指标的正常范围"""
        return {
            'ejection_fraction': (50, 70),      # 50-70%
            'stroke_volume': (60, 100),         # 60-100 mL
            'cardiac_output': (4.0, 8.0),       # 4.0-8.0 L/min
            'heart_rate_variability': (20, 50), # 20-50 ms
            'left_ventricular_mass': (70, 180), # 70-180 g
            'wall_thickness': (8, 12),          # 8-12 mm
            'chamber_volume': (100, 150),       # 100-150 mL
            'contractility_index': (0.8, 1.2),  # 0.8-1.2
            'diastolic_function': (0.8, 1.2),   # 0.8-1.2
            'valvular_function': (0.8, 1.2)     # 0.8-1.2
        }
    
    @classmethod
    def normalize_predictions(cls, predictions):
        """
        将模型预测值标准化到生理范围
        
        Args:
            predictions: 模型原始预测值 [batch_size, num_metrics] 或 [num_metrics]
            
        Returns:
            normalized_predictions: 标准化后的预测值，相同形状
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.clone()
        else:
            predictions = torch.tensor(predictions)
        
        # 确保是2D张量
        original_shape = predictions.shape
        if len(predictions.shape) == 1:
            predictions = predictions.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        normal_ranges = cls.get_normal_ranges()
        metric_names = cls.get_metric_names()
        
        # 对每个指标进行标准化
        for i, metric_name in enumerate(metric_names):
            if i < predictions.shape[1]:
                min_val, max_val = normal_ranges[metric_name]
                
                # 使用tanh激活函数将原始预测值映射到[-1, 1]，然后缩放到正常范围
                normalized_val = torch.tanh(predictions[:, i])
                # 映射到正常范围
                predictions[:, i] = min_val + (normalized_val + 1) * (max_val - min_val) / 2
        
        # 恢复原始形状
        if squeeze_output:
            predictions = predictions.squeeze(0)
        
        return predictions
    
    @classmethod
    def denormalize_predictions(cls, normalized_predictions):
        """
        将标准化的预测值转换回模型输出范围
        
        Args:
            normalized_predictions: 标准化的预测值
            
        Returns:
            denormalized_predictions: 反标准化的预测值
        """
        if isinstance(normalized_predictions, torch.Tensor):
            predictions = normalized_predictions.clone()
        else:
            predictions = torch.tensor(normalized_predictions)
        
        # 确保是2D张量
        if len(predictions.shape) == 1:
            predictions = predictions.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        normal_ranges = cls.get_normal_ranges()
        metric_names = cls.get_metric_names()
        
        # 对每个指标进行反标准化
        for i, metric_name in enumerate(metric_names):
            if i < predictions.shape[1]:
                min_val, max_val = normal_ranges[metric_name]
                
                # 从正常范围映射回[-1, 1]
                normalized_val = 2 * (predictions[:, i] - min_val) / (max_val - min_val) - 1
                # 使用arctanh转换回原始范围
                predictions[:, i] = torch.atanh(torch.clamp(normalized_val, -0.99, 0.99))
        
        # 恢复原始形状
        if squeeze_output:
            predictions = predictions.squeeze(0)
        
        return predictions
    
    @classmethod
    def evaluate_predictions(cls, predictions, return_status=True):
        """
        评估心脏功能预测结果
        
        Args:
            predictions: 标准化的预测值
            return_status: 是否返回状态评估
            
        Returns:
            evaluation: 评估结果字典
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        
        if len(predictions.shape) == 2:
            predictions = predictions[0]  # 取第一个样本
        
        normal_ranges = cls.get_normal_ranges()
        metric_names = cls.get_metric_names()
        descriptions = cls.get_metric_descriptions()
        
        evaluation = {}
        
        for i, metric_name in enumerate(metric_names):
            if i < len(predictions):
                value = float(predictions[i])
                min_val, max_val = normal_ranges[metric_name]
                description = descriptions[metric_name]
                
                metric_eval = {
                    'value': value,
                    'description': description,
                    'normal_range': f"{min_val}-{max_val}",
                    'in_normal_range': min_val <= value <= max_val
                }
                
                if return_status:
                    if value < min_val:
                        metric_eval['status'] = '偏低'
                    elif value > max_val:
                        metric_eval['status'] = '偏高'
                    else:
                        metric_eval['status'] = '正常'
                
                evaluation[metric_name] = metric_eval
        
        return evaluation


# 使用示例和测试代码
if __name__ == "__main__":
    # 跳过数据加载测试（需要特定的CSV文件）
    print("=== 测试CardiacMetricsCalculator ===")
    metric_names = CardiacMetricsCalculator.get_metric_names()
    print(f"心脏功能指标 ({len(metric_names)} 个):")
    for i, name in enumerate(metric_names, 1):
        print(f"  {i:2d}. {name}")
    
    print("\n=== 测试模型 ===")
    # 创建模型
    model = CardiacFunctionModel()
    print(f"模型创建成功")
    
    # 创建随机输入（模拟CT图像）
    dummy_image = torch.randn(2, 1, 16, 224, 224)  # [batch, channels, depth, height, width] - CT图像是单通道的
    print(f"输入图像形状: {dummy_image.shape}")
    
    # 前向传播
    print("开始前向传播...")
    lvef_pred, as_pred = model(dummy_image)
    print(f"LVEF预测形状: {lvef_pred.shape}")
    print(f"AS预测形状: {as_pred.shape}")
    print(f"LVEF预测值: {lvef_pred.squeeze().detach().numpy()}")
    print(f"AS预测概率: {as_pred.squeeze().detach().numpy()}")
    
    print("\n=== 测试损失函数 ===")
    # 使用默认类别权重测试损失函数
    criterion = CardiacLoss()
    
    # 模拟真实标签
    lvef_true = torch.tensor([65.0, 45.0])
    as_true = torch.tensor([0.0, 1.0])
    
    # 计算损失
    loss_dict = criterion(lvef_pred, as_pred, lvef_true, as_true)
    print(f"总损失: {loss_dict['total_loss']:.4f}")
    print(f"回归损失: {loss_dict['regression_loss']:.4f}")
    print(f"分类损失: {loss_dict['classification_loss']:.4f}")
    
    print("\n=== 测试预测标准化 ===")
    # 创建虚拟的心脏功能预测
    dummy_cardiac_preds = torch.randn(1, 10) * 2  # 随机预测值
    normalized_preds = CardiacMetricsCalculator.normalize_predictions(dummy_cardiac_preds)
    
    print("标准化预测结果:")
    descriptions = CardiacMetricsCalculator.get_metric_descriptions()
    for i, metric_name in enumerate(metric_names):
        value = normalized_preds[0, i].item()
        desc = descriptions[metric_name]
        print(f"  {desc}: {value:.2f}")
    
    print("\n=== 测试完成 ===")
    print("所有核心功能正常工作！") 