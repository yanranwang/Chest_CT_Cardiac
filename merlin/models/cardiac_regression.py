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
        
        # if提供了预Trainweightspath，Loadweights
        if pretrained_path:
            self._load_pretrained_weights(pretrained_path)
    
    def _load_pretrained_weights(self, path):
        """Load预训练的Merlin权重"""
        try:
            state_dict = torch.load(path, map_location='cpu')
            
            # TODO: Translate '只'Loadimageencode器相关的weights
            image_encoder_dict = {}
            
            # TODO: Translate '遍历所'hasweightskey
            for key, value in state_dict.items():
                # findencode_image.i3_resnet开头的weightskey
                if key.startswith('encode_image.i3_resnet.'):
                    # Removeencode_image.i3_resnet.前缀，只保留实际的层名
                    new_key = key.replace('encode_image.i3_resnet.', '')
                    image_encoder_dict[new_key] = value
                    print(f"映射权重: {key} -> {new_key}")
            
            if image_encoder_dict:
                # TODO: Translate '尝试'Loadweightstoi3_resnet
                missing_keys, unexpected_keys = self.i3_resnet.load_state_dict(image_encoder_dict, strict=False)
                
                if missing_keys:
                    print(f"缺少的权重键: {missing_keys}")
                if unexpected_keys:
                    print(f"意外的权重键: {unexpected_keys}")
                    
                print(f"成功Load预训练的图像编码器权重，共 {len(image_encoder_dict)} 个参数")
            else:
                print("警告: 在预训练权重中未找到图像编码器相关的权重")
                print("可用的权重键:")
                for key in list(state_dict.keys())[:10]:  # show前10个key
                    print(f"  {key}")
                if len(state_dict.keys()) > 10:
                    print(f"  ... 还有 {len(state_dict.keys()) - 10} 个键")
                
        except Exception as e:
            print(f"Load预训练权重失败: {e}")
            print("将使用随机Initialize的权重继续训练")
    
    def forward(self, image):
        contrastive_features, ehr_features = self.i3_resnet(image)
        return contrastive_features, ehr_features


class CardiacPredictionHead(nn.Module):
    """心脏功能预测头：LVEF回归 + AS二分类"""
    def __init__(self, input_dim=512, hidden_dims=[256, 128]):
        super().__init__()
        
        # TODO: Translate '共享'featuresExtract层
        shared_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            shared_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.LayerNorm(hidden_dim)  # TODO: Translate '使用'LayerNorm替代BatchNorm，not依赖于batch_size
            ])
            prev_dim = hidden_dim
        
        self.shared_features = nn.Sequential(*shared_layers)
        
        # LVEFregression头
        self.lvef_regressor = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)  # LVEFOutput (射血分数百分比)
        )
        
        # AS二classification头
        self.as_classifier = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)  # AS存in性Output (notin这里加sigmoid，inforward中Process)
        )
        
        # Initializeweights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize网络权重"""
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
        
        # standard化features
        features = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
        
        # TODO: Translate '共享'featuresExtract
        shared_feat = self.shared_features(features)
        
        # LVEFregressionPredict
        lvef_pred = self.lvef_regressor(shared_feat)
        
        # ASclassificationPredict - 使用stable的sigmoid激活
        as_logits = self.as_classifier(shared_feat)
        as_pred = torch.sigmoid(torch.clamp(as_logits, min=-10, max=10))
        
        return lvef_pred, as_pred


class CardiacFunctionModel(nn.Module):
    """完整的心脏功能预测模型：LVEF回归 + AS二分类"""
    def __init__(self, pretrained_model_path=None):
        super().__init__()
        
        # imageencode器
        self.image_encoder = CardiacImageEncoder(pretrained_model_path)
        
        # cardiac functionPredict头
        self.cardiac_predictor = CardiacPredictionHead(input_dim=512)
        
        # is否冻结imageencode器
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
            return_features: 是否Return中间特征
        
        Returns:
            lvef_pred: LVEF预测值 [batch_size, 1]
            as_pred: AS存在性概率 [batch_size, 1]  
            image_features: 图像特征 [batch_size, 512] (可选)
        """
        # Extractimagefeatures
        image_features, _ = self.image_encoder(image)
        
        # cardiac function prediction
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
            split: 数据Split ('train', 'val', 'test')
            test_size: 测试集比例
            random_state: 随机种子
        """
        self.csv_path = csv_path
        self.image_dir = image_dir
        
        # ReadCSVdata
        self.df = pd.read_csv(csv_path)
        
        # Filterhas效data（同时hasLVEFandAS_definitelabels）
        valid_mask = self.df['lvef'].notna() & self.df['AS_definite'].notna()
        self.df = self.df[valid_mask].reset_index(drop=True)
        
        print(f"有效Sample数量: {len(self.df)}")
        print(f"LVEFRange: {self.df['lvef'].min():.2f} - {self.df['lvef'].max():.2f}")
        print(f"AS分布: {self.df['AS_definite'].value_counts().to_dict()}")
        
        # dataSplit
        if split in ['train', 'val']:
            train_indices, val_indices = train_test_split(
                range(len(self.df)), 
                test_size=test_size, 
                random_state=random_state,
                stratify=self.df['AS_definite']  # TODO: Translate '保持'ASclass别平衡
            )
            
            if split == 'train':
                self.indices = train_indices
            else:
                self.indices = val_indices
        else:
            self.indices = list(range(len(self.df)))
        
        print(f"{split}集Sample数量: {len(self.indices)}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """Get单samples"""
        real_idx = self.indices[idx]
        row = self.df.iloc[real_idx]
        
        # Getlabels
        lvef = float(row['lvef'])
        as_definite = int(row['AS_definite'])
        
        # Getimage标识符
        mrn = row['mrn']
        accession_number = row['accession_number']
        
        sample = {
            'mrn': mrn,
            'accession_number': accession_number,
            'lvef': torch.tensor(lvef, dtype=torch.float32),
            'as_definite': torch.tensor(as_definite, dtype=torch.float32),
            'lvef_normalized': torch.tensor((lvef - 59.22) / 11.85, dtype=torch.float32),  # standard化LVEF
        }
        
        # ifhasimage目录，可以in这里LoadCTimage
        # if self.image_dir:
        #     image_path = os.path.join(self.image_dir, f"{mrn}_{accession_number}.nii.gz")
        #     image = self.load_ct_image(image_path)
        #     sample['image'] = image
        
        return sample
    
    def get_class_weights(self):
        """CalculateAS分类的Class权重，用于ProcessClass不平衡"""
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
        
        # regressionloss：均方误差
        self.regression_loss = nn.MSELoss()
        
        # classificationloss：始终使用BCELoss，避免BCEWithLogitsLoss的冲突
        # TODO: Translate '因为'ASclassification器Output已经经过sigmoid激活
        self.classification_loss = nn.BCELoss()
        
        # TODO: Translate '存储'class别weights以供manual应用
        self.class_weights = class_weights
    
    def forward(self, lvef_pred, as_pred, lvef_true, as_true):
        """
        Args:
            lvef_pred: LVEF预测值 [batch_size, 1]
            as_pred: AS预测概率 [batch_size, 1] (已经过sigmoid激活)
            lvef_true: LVEF真实值 [batch_size]
            as_true: AS真实标签 [batch_size]
        """
        # LVEFregressionloss
        reg_loss = self.regression_loss(lvef_pred.squeeze(), lvef_true)
        
        # ASclassificationloss - Add数valuestable性保护
        as_pred_clamped = torch.clamp(as_pred.squeeze(), min=1e-7, max=1.0 - 1e-7)
        as_true_clamped = torch.clamp(as_true, min=0.0, max=1.0)
        
        # Calculate基础BCEloss
        clf_loss = self.classification_loss(as_pred_clamped, as_true_clamped)
        
        # if提供了class别weights，manual应用weights
        if self.class_weights is not None:
            # Calculate正classand负class的weights
            pos_weight = self.class_weights[1] if len(self.class_weights) > 1 else 1.0
            neg_weight = self.class_weights[0] if len(self.class_weights) > 0 else 1.0
            
            # TODO: Translate '应用'weights
            weights = as_true_clamped * pos_weight + (1 - as_true_clamped) * neg_weight
            clf_loss = clf_loss * weights.mean()
        
        # combinationloss
        total_loss = (self.regression_weight * reg_loss + 
                     self.classification_weight * clf_loss)
        
        return {
            'total_loss': total_loss,
            'regression_loss': reg_loss,
            'classification_loss': clf_loss
        }


def create_cardiac_dataloaders(csv_path, batch_size=32, num_workers=4):
    """Create训练和Validate数据Load器"""
    
    # Createdata集
    train_dataset = CardiacDataset(csv_path, split='train')
    val_dataset = CardiacDataset(csv_path, split='val')
    
    # Getclass别weights
    class_weights = train_dataset.get_class_weights()
    print(f"ASClass权重: {class_weights}")
    
    # CreatedataLoad器
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
    """心脏功能指标Calculate和标准化工具"""
    
    @staticmethod
    def get_metric_names():
        """Get心脏功能指标名称列表"""
        return [
            'ejection_fraction',       # TODO: Translate '射血分数
            ''stroke_volume',          # TODO: Translate '每搏'Output量
            'cardiac_output',         # TODO: Translate '心'Output量
            'heart_rate_variability', # TODO: Translate '心率变异性
            ''left_ventricular_mass',  # TODO: Translate '左心室'mass
            'wall_thickness',         # TODO: Translate '室壁厚度
            ''chamber_volume',         # TODO: Translate '心室容积
            ''contractility_index',    # shrink性指数
            'diastolic_function',     # TODO: Translate '舒张功能
            ''valvular_function'       # TODO: Translate '瓣膜功能
        ]
    
    @'staticmethod
    def get_metric_descriptions():
        """Get心脏功能指标的中文描述"""
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
        """Get心脏功能指标的正常Range"""
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
        将模型预测值标准化到生理Range
        
        Args:
            predictions: 模型原始预测值 [batch_size, num_metrics] 或 [num_metrics]
            
        Returns:
            normalized_predictions: 标准化后的预测值，相同形状
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.clone()
        else:
            predictions = torch.tensor(predictions)
        
        # TODO: Translate '确保'is2D张量
        original_shape = predictions.shape
        if len(predictions.shape) == 1:
            predictions = predictions.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        normal_ranges = cls.get_normal_ranges()
        metric_names = cls.get_metric_names()
        
        # TODO: Translate '对'each指标进行standard化
        for i, metric_name in enumerate(metric_names):
            if i < predictions.shape[1]:
                min_val, max_val = normal_ranges[metric_name]
                
                # TODO: Translate '使用'tanh激活function将原始Predictvaluemappingto[-1, 1]，thenscaleto正常range
                normalized_val = torch.tanh(predictions[:, i])
                # mappingto正常range
                predictions[:, i] = min_val + (normalized_val + 1) * (max_val - min_val) / 2
        
        # restore原始shape
        if squeeze_output:
            predictions = predictions.squeeze(0)
        
        return predictions
    
    @classmethod
    def denormalize_predictions(cls, normalized_predictions):
        """
        将标准化的预测值Convert回模型输出Range
        
        Args:
            normalized_predictions: 标准化的预测值
            
        Returns:
            denormalized_predictions: 反标准化的预测值
        """
        if isinstance(normalized_predictions, torch.Tensor):
            predictions = normalized_predictions.clone()
        else:
            predictions = torch.tensor(normalized_predictions)
        
        # TODO: Translate '确保'is2D张量
        if len(predictions.shape) == 1:
            predictions = predictions.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        normal_ranges = cls.get_normal_ranges()
        metric_names = cls.get_metric_names()
        
        # TODO: Translate '对'each指标进行反standard化
        for i, metric_name in enumerate(metric_names):
            if i < predictions.shape[1]:
                min_val, max_val = normal_ranges[metric_name]
                
                # from正常rangemapping回[-1, 1]
                normalized_val = 2 * (predictions[:, i] - min_val) / (max_val - min_val) - 1
                # TODO: Translate '使用'arctanhConvert回原始range
                predictions[:, i] = torch.atanh(torch.clamp(normalized_val, -0.99, 0.99))
        
        # restore原始shape
        if squeeze_output:
            predictions = predictions.squeeze(0)
        
        return predictions
    
    @classmethod
    def evaluate_predictions(cls, predictions, return_status=True):
        """
        评估心脏功能预测结果
        
        Args:
            predictions: 标准化的预测值
            return_status: 是否Return状态评估
            
        Returns:
            evaluation: 评估结果字典
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        
        if len(predictions.shape) == 2:
            predictions = predictions[0]  # TODO: Translate '取'firstsample
        
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


# TODO: Translate '使用示例'andTest代码
if __name__ == "__main__":
    # skipdataLoadTest（需要特定的CSVfile）
    print("=== 测试CardiacMetricsCalculator ===")
    metric_names = CardiacMetricsCalculator.get_metric_names()
    print(f"心脏功能指标 ({len(metric_names)} 个):")
    for i, name in enumerate(metric_names, 1):
        print(f"  {i:2d}. {name}")
    
    print("\n=== 测试模型 ===")
    # Createmodel
    model = CardiacFunctionModel()
    print(f"模型Create成功")
    
    # CreaterandomInput（模拟CTimage）
    dummy_image = torch.randn(2, 1, 16, 224, 224)  # [batch, channels, depth, height, width] - CTimageis单通道的
    print(f"输入图像形状: {dummy_image.shape}")
    
    # TODO: Translate '前向传播'print("开始前向传播...")
    lvef_pred, as_pred = model(dummy_image)
    print(f"LVEF预测形状: {lvef_pred.shape}")
    print(f"AS预测形状: {as_pred.shape}")
    print(f"LVEF预测值: {lvef_pred.squeeze().detach().numpy()}")
    print(f"AS预测概率: {as_pred.squeeze().detach().numpy()}")
    
    print("\n=== 测试损失函数 ===")
    # TODO: Translate '使用'defaultclass别weightsTestlossfunction
    criterion = CardiacLoss()
    
    # TODO: Translate '模拟真实'labels
    lvef_true = torch.tensor([65.0, 45.0])
    as_true = torch.tensor([0.0, 1.0])
    
    # Calculateloss
    loss_dict = criterion(lvef_pred, as_pred, lvef_true, as_true)
    print(f"总损失: {loss_dict['total_loss']:.4f}")
    print(f"回归损失: {loss_dict['regression_loss']:.4f}")
    print(f"分类损失: {loss_dict['classification_loss']:.4f}")
    
    print("\n=== 测试预测标准化 ===")
    # Create虚拟的cardiac functionPredict
    dummy_cardiac_preds = torch.randn(1, 10) * 2  # randomPredictvalue
    normalized_preds = CardiacMetricsCalculator.normalize_predictions(dummy_cardiac_preds)
    
    print("标准化预测结果:")
    descriptions = CardiacMetricsCalculator.get_metric_descriptions()
    for i, metric_name in enumerate(metric_names):
        value = normalized_preds[0, i].item()
        desc = descriptions[metric_name]
        print(f"  {desc}: {value:.2f}")
    
    print("\n=== 测试完成 ===")
    print("所有核心功能正常工作！") 