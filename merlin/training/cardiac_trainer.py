import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import monai
from monai.losses import DiceLoss
from monai.transforms import Compose
import logging
from pathlib import Path
from tqdm import tqdm
import warnings

# 抑制不必要的警告
warnings.filterwarnings('ignore', category=UserWarning)

from merlin.models.cardiac_regression import CardiacFunctionModel, CardiacMetricsCalculator
from merlin.data.dataloaders import CTPersistentDataset
from merlin.data.monai_transforms import ImageTransforms


class CardiacDataset(Dataset):
    """心脏功能数据集 - 支持从CSV文件读取数据"""
    def __init__(self, data_list, transform=None, cardiac_metric_columns=None):
        self.data_list = data_list
        self.transform = transform or ImageTransforms
        self.cardiac_metric_columns = cardiac_metric_columns or []
        self.metric_names = CardiacMetricsCalculator.get_metric_names()
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        sample = self.data_list[idx]
        
        # 加载和预处理图像
        if self.transform:
            sample = self.transform(sample)
        
        # 获取心脏功能标签
        if 'cardiac_metrics' in sample and sample['cardiac_metrics'] is not None:
            cardiac_metrics = torch.tensor(sample['cardiac_metrics'], dtype=torch.float32)
        else:
            # 如果没有真实标签，生成模拟数据用于演示
            cardiac_metrics = torch.tensor(self._generate_dummy_labels(), dtype=torch.float32)
        
        # 确保标签包含LVEF和AS两个值
        if len(cardiac_metrics) < 2:
            # 如果标签数量不足，使用模拟数据
            cardiac_metrics = torch.tensor(self._generate_dummy_labels(), dtype=torch.float32)
        
        return {
            'image': sample['image'],
            'cardiac_metrics': cardiac_metrics,
            'patient_id': sample.get('patient_id', f'patient_{idx}'),
            'basename': sample.get('basename', f'unknown_{idx}'),
            'folder': sample.get('folder', 'unknown'),
            'image_path': sample.get('image', ''),
            'metadata': sample.get('metadata', {})
        }
    
    def _generate_dummy_labels(self):
        """生成模拟的心脏功能标签（用于演示）"""
        # 生成LVEF和AS的模拟数据
        # LVEF: 正常范围约50-70%，这里生成标准化值
        lvef = np.float32(np.random.normal(0, 1))
        # AS: 二分类，0或1
        as_label = np.float32(np.random.randint(0, 2))
        
        return np.array([lvef, as_label], dtype=np.float32)


class CardiacTrainer:
    """心脏功能回归训练器"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # 创建输出目录
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志（必须在使用logger之前）
        self._setup_logging()
        
        # 设置随机种子
        self._set_random_seed(config.get('seed', 42))
        
        # 初始化模型
        self.model = self._build_model()
        
        # 初始化优化器和调度器
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # 初始化损失函数
        self.criterion = self._build_loss_function()
        
        # 初始化训练记录
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.epoch_times = []
        
        # 初始化tensorboard记录器
        if config.get('use_tensorboard', True):
            self.writer = SummaryWriter(self.output_dir / 'tensorboard')
        else:
            self.writer = None
    
    def _set_random_seed(self, seed):
        """设置随机种子"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    
    def _build_model(self):
        """构建模型"""
        model = CardiacFunctionModel(
            pretrained_model_path=self.config.get('pretrained_model_path')
        )
        
        # 是否冻结图像编码器
        if self.config.get('freeze_encoder', False):
            model.freeze_encoder(True)
            self.logger.info("图像编码器已冻结")
        
        model = model.to(self.device)
        
        # 多GPU训练
        if torch.cuda.device_count() > 1:
            # 检查batch_size是否足够大，避免每个GPU分到的batch_size为1
            batch_size = self.config.get('batch_size', 4)
            num_gpus = torch.cuda.device_count()
            if batch_size < num_gpus:
                self.logger.warning(f"批量大小 {batch_size} 小于GPU数量 {num_gpus}，可能导致BatchNorm错误")
                self.logger.warning("建议增加批量大小或减少GPU数量")
            
            model = nn.DataParallel(model)
            self.logger.info(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
        
        return model
    
    def _build_optimizer(self):
        """构建优化器"""
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        lr = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 1e-5)
        
        if optimizer_name == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}")
        
        return optimizer
    
    def _build_scheduler(self):
        """构建学习率调度器"""
        scheduler_config = self.config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'cosine')
        
        if scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config.get('epochs', 100),
                eta_min=scheduler_config.get('eta_min', 1e-6)
            )
        elif scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_type == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 10)
            )
        else:
            scheduler = None
        
        return scheduler
    
    def _build_loss_function(self):
        """构建损失函数"""
        from merlin.models.cardiac_regression import CardiacLoss
        
        # 使用专门的心脏功能损失函数，处理LVEF回归和AS分类
        regression_weight = self.config.get('regression_weight', 1.0)
        classification_weight = self.config.get('classification_weight', 1.0)
        
        criterion = CardiacLoss(
            regression_weight=regression_weight,
            classification_weight=classification_weight
        )
        
        return criterion
    
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _format_time(self, seconds):
        """格式化时间显示"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes:.0f}m {secs:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h {minutes:.0f}m"
    
    def train_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(train_loader)
        
        # 创建进度条
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1:3d}/{self.config.get("epochs", 100)}', 
                   ncols=120, leave=False)
        
        batch_losses = []
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            
            # 从cardiac_metrics中提取LVEF和AS目标值
            cardiac_metrics = batch['cardiac_metrics'].to(self.device)
            # 假设cardiac_metrics的第一个值是LVEF，第二个值是AS
            if cardiac_metrics.shape[1] >= 2:
                lvef_targets = cardiac_metrics[:, 0]  # LVEF目标值
                as_targets = cardiac_metrics[:, 1]    # AS目标值
            else:
                # 如果只有一个值，假设是LVEF，AS设为0
                lvef_targets = cardiac_metrics[:, 0]
                as_targets = torch.zeros_like(lvef_targets)
            
            # 前向传播
            self.optimizer.zero_grad()
            lvef_preds, as_preds = self.model(images)
            
            # 计算损失
            loss_dict = self.criterion(lvef_preds, as_preds, lvef_targets, as_targets)
            loss = loss_dict['total_loss']
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if self.config.get('grad_clip', None):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            
            self.optimizer.step()
            
            epoch_loss += loss.item()
            batch_losses.append(loss.item())
            
            # 更新进度条
            if len(batch_losses) >= 10:  # 显示最近10个batch的平均损失
                recent_loss = np.mean(batch_losses[-10:])
            else:
                recent_loss = np.mean(batch_losses)
            
            pbar.set_postfix({
                'Loss': f'{recent_loss:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                'GPU': f'{torch.cuda.get_device_name(0)[:12]}' if torch.cuda.is_available() else 'CPU'
            })
            
            # 详细日志记录
            if batch_idx % max(1, self.config.get('log_interval', 10)) == 0:
                self.logger.info(
                    f'Epoch {epoch+1:3d} [{batch_idx:4d}/{num_batches:4d}] '
                    f'Loss: {loss.item():.6f} | LR: {self.optimizer.param_groups[0]["lr"]:.2e} | '
                    f'GPU Memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB' 
                    if torch.cuda.is_available() else ''
                )
                
                if self.writer:
                    step = epoch * num_batches + batch_idx
                    self.writer.add_scalar('Train/BatchLoss', loss.item(), step)
                    self.writer.add_scalar('Train/LearningRate', self.optimizer.param_groups[0]['lr'], step)
        
        avg_loss = epoch_loss / num_batches
        self.train_losses.append(avg_loss)
        
        # 清除进度条
        pbar.close()
        
        return avg_loss
    
    def validate_epoch(self, val_loader, epoch):
        """验证一个epoch"""
        self.model.eval()
        epoch_loss = 0.0
        all_predictions = []
        all_targets = []
        
        # 创建验证进度条
        pbar = tqdm(val_loader, desc=f'Validating', ncols=120, leave=False)
        
        with torch.no_grad():
            for batch in pbar:
                images = batch['image'].to(self.device)
                
                # 从cardiac_metrics中提取LVEF和AS目标值
                cardiac_metrics = batch['cardiac_metrics'].to(self.device)
                if cardiac_metrics.shape[1] >= 2:
                    lvef_targets = cardiac_metrics[:, 0]  # LVEF目标值
                    as_targets = cardiac_metrics[:, 1]    # AS目标值
                else:
                    lvef_targets = cardiac_metrics[:, 0]
                    as_targets = torch.zeros_like(lvef_targets)
                
                # 前向传播
                lvef_preds, as_preds = self.model(images)
                
                # 计算损失
                loss_dict = self.criterion(lvef_preds, as_preds, lvef_targets, as_targets)
                loss = loss_dict['total_loss']
                epoch_loss += loss.item()
                
                # 更新验证进度条
                pbar.set_postfix({'Val Loss': f'{loss.item():.4f}'})
                
                # 收集预测和真值用于指标计算
                predictions = torch.stack([lvef_preds.squeeze(), as_preds.squeeze()], dim=1)
                targets = torch.stack([lvef_targets, as_targets], dim=1)
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        pbar.close()
        
        avg_loss = epoch_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        
        # 计算评估指标
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        metrics = self._calculate_metrics(all_predictions, all_targets)
        
        # 记录验证结果
        self.logger.info(f'验证结果 - Epoch {epoch+1:3d} | Loss: {avg_loss:.6f}')
        for metric_name, value in metrics.items():
            self.logger.info(f'  {metric_name}: {value:.4f}')
        
        if self.writer:
            self.writer.add_scalar('Val/Loss', avg_loss, epoch)
            for metric_name, value in metrics.items():
                self.writer.add_scalar(f'Val/{metric_name}', value, epoch)
        
        return avg_loss, metrics
    
    def _calculate_metrics(self, predictions, targets):
        """计算评估指标"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {}
        
        # LVEF回归指标（第0列）
        lvef_preds = predictions[:, 0]
        lvef_targets = targets[:, 0]
        
        metrics['LVEF_MSE'] = mean_squared_error(lvef_targets, lvef_preds)
        metrics['LVEF_MAE'] = mean_absolute_error(lvef_targets, lvef_preds)
        metrics['LVEF_R2'] = r2_score(lvef_targets, lvef_preds)
        
        # AS分类指标（第1列）
        as_preds = predictions[:, 1]
        as_targets = targets[:, 1]
        
        # 将概率转换为二分类预测
        as_pred_binary = (as_preds > 0.5).astype(int)
        as_targets_binary = as_targets.astype(int)
        
        metrics['AS_Accuracy'] = accuracy_score(as_targets_binary, as_pred_binary)
        metrics['AS_Precision'] = precision_score(as_targets_binary, as_pred_binary, zero_division=0)
        metrics['AS_Recall'] = recall_score(as_targets_binary, as_pred_binary, zero_division=0)
        metrics['AS_F1'] = f1_score(as_targets_binary, as_pred_binary, zero_division=0)
        
        # 计算AS的AUC
        try:
            from sklearn.metrics import roc_auc_score
            metrics['AS_AUC'] = roc_auc_score(as_targets_binary, as_preds)
        except ValueError:
            metrics['AS_AUC'] = 0.0
        
        return metrics
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # 保存最新检查点
        torch.save(checkpoint, self.output_dir / 'checkpoint_latest.pth')
        
        # 保存最佳模型
        if is_best:
            torch.save(checkpoint, self.output_dir / 'checkpoint_best.pth')
            torch.save(self.model.state_dict(), self.output_dir / 'best_model.pth')
            self.logger.info(f'💾 保存最佳模型 (Epoch {epoch+1}, Val Loss: {self.best_val_loss:.6f})')
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        self.logger.info(f'从 {checkpoint_path} 加载检查点')
        return checkpoint['epoch']
    
    def train(self, train_loader, val_loader=None, start_epoch=0):
        """完整训练流程"""
        epochs = self.config.get('epochs', 100)
        
        # 打印训练开始信息
        print("=" * 80)
        print(f"🚀 开始训练心脏功能预测模型")
        print("=" * 80)
        print(f"📊 训练参数:")
        print(f"   总轮数: {epochs}")
        print(f"   批量大小: {self.config.get('batch_size', 4)}")
        print(f"   学习率: {self.config.get('learning_rate', 1e-4)}")
        print(f"   设备: {self.device}")
        print(f"   优化器: {self.config.get('optimizer', 'adam')}")
        print(f"   调度器: {self.config.get('scheduler', {}).get('type', 'None')}")
        print(f"📁 数据统计:")
        print(f"   训练集大小: {len(train_loader.dataset)}")
        if val_loader:
            print(f"   验证集大小: {len(val_loader.dataset)}")
        print(f"   每轮批次数: {len(train_loader)}")
        print(f"💾 输出目录: {self.output_dir}")
        if self.writer:
            print(f"📈 TensorBoard: {self.output_dir / 'tensorboard'}")
        print("=" * 80)
        
        self.logger.info(f'开始训练，共 {epochs} 个epoch')
        
        # 训练开始时间
        training_start_time = time.time()
        
        for epoch in range(start_epoch, epochs):
            epoch_start_time = time.time()
            
            # 打印epoch开始信息
            print(f"\n🔄 Epoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # 训练
            train_loss = self.train_epoch(train_loader, epoch)
            
            # 验证
            val_loss, val_metrics = None, {}
            if val_loader:
                val_loss, val_metrics = self.validate_epoch(val_loader, epoch)
                
                # 更新学习率调度器
                if self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                
                # 检查是否为最佳模型
                is_best = val_loss < self.best_val_loss
                if is_best:
                    improvement = self.best_val_loss - val_loss
                    self.best_val_loss = val_loss
                    print(f"✨ 新的最佳模型！验证损失降低了 {improvement:.6f}")
                
                # 保存检查点
                if epoch % self.config.get('save_interval', 10) == 0 or is_best:
                    self.save_checkpoint(epoch, is_best)
            else:
                # 无验证集时
                if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
                
                if epoch % self.config.get('save_interval', 10) == 0:
                    self.save_checkpoint(epoch)
            
            # 计算时间
            epoch_time = time.time() - epoch_start_time
            self.epoch_times.append(epoch_time)
            
            # 预估剩余时间
            avg_epoch_time = np.mean(self.epoch_times[-10:])  # 使用最近10个epoch的平均时间
            remaining_epochs = epochs - epoch - 1
            estimated_remaining_time = avg_epoch_time * remaining_epochs
            
            # 打印epoch总结
            print(f"📊 Epoch {epoch+1} 总结:")
            print(f"   训练损失: {train_loss:.6f}")
            if val_loss is not None:
                print(f"   验证损失: {val_loss:.6f}")
                print(f"   LVEF R²: {val_metrics.get('LVEF_R2', 0):.4f}")
                print(f"   AS 准确率: {val_metrics.get('AS_Accuracy', 0):.4f}")
            print(f"   轮次耗时: {self._format_time(epoch_time)}")
            print(f"   平均耗时: {self._format_time(avg_epoch_time)}")
            print(f"   预估剩余: {self._format_time(estimated_remaining_time)}")
            print(f"   当前学习率: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # 记录到tensorboard
            if self.writer:
                self.writer.add_scalar('Train/EpochLoss', train_loss, epoch)
                self.writer.add_scalar('Train/EpochTime', epoch_time, epoch)
                if val_loader:
                    self.writer.add_scalar('Val/EpochLoss', val_loss, epoch)
        
        # 训练完成
        total_training_time = time.time() - training_start_time
        
        print("\n" + "=" * 80)
        print("🎉 训练完成！")
        print("=" * 80)
        print(f"📈 训练统计:")
        print(f"   总训练时间: {self._format_time(total_training_time)}")
        print(f"   平均每轮时间: {self._format_time(np.mean(self.epoch_times))}")
        print(f"   最佳验证损失: {self.best_val_loss:.6f}")
        print(f"💾 输出文件:")
        print(f"   最佳模型: {self.output_dir}/best_model.pth")
        print(f"   训练日志: {self.output_dir}/training.log")
        print(f"   配置文件: {self.output_dir}/config.json")
        if self.writer:
            print(f"   TensorBoard: {self.output_dir}/tensorboard")
        print("=" * 80)
        
        self.logger.info('训练完成！')
        
        # 保存最终模型
        self.save_checkpoint(epochs - 1)
        
        # 保存训练配置
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        if self.writer:
            self.writer.close()


def load_and_validate_csv_data(config):
    """加载和验证CSV数据"""
    csv_path = config.get('csv_path')
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
    
    print(f"从 {csv_path} 读取数据...")
    df = pd.read_csv(csv_path)
    print(f"原始数据集大小: {len(df)} 行")
    
    # 检查必需的列
    required_columns = config.get('required_columns', ['basename', 'folder'])
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"CSV文件中缺少必需的列: {missing_columns}")
    
    # 检查心脏功能指标列
    cardiac_metric_columns = config.get('cardiac_metric_columns', [])
    if cardiac_metric_columns:
        missing_cardiac_columns = [col for col in cardiac_metric_columns if col not in df.columns]
        if missing_cardiac_columns:
            print(f"警告: CSV文件中缺少心脏功能指标列: {missing_cardiac_columns}")
            cardiac_metric_columns = [col for col in cardiac_metric_columns if col in df.columns]
        if cardiac_metric_columns:
            print(f"找到心脏功能指标列: {cardiac_metric_columns}")
        else:
            print("注意: 未找到任何心脏功能指标数据，将使用模拟标签进行训练演示")
    else:
        print("注意: 配置中未指定心脏功能指标列，将使用模拟标签进行训练演示")
    
    # 数据清理
    if config.get('remove_missing_files', True):
        initial_count = len(df)
        df = df.dropna(subset=['basename', 'folder'])
        df = df[df['basename'].notna() & df['folder'].notna()]
        if len(df) < initial_count:
            print(f"移除了 {initial_count - len(df)} 行缺失basename或folder的数据")
    
    # 移除重复项
    if config.get('remove_duplicates', True):
        initial_count = len(df)
        df = df.drop_duplicates(subset=['basename', 'folder'])
        if len(df) < initial_count:
            print(f"移除了 {initial_count - len(df)} 行重复数据")
    
    print(f"清理后数据集大小: {len(df)} 行")
    return df, cardiac_metric_columns


def build_data_list(df, config, cardiac_metric_columns):
    """从DataFrame构建数据列表"""
    base_path = config.get('base_path', '/dataNAS/data/ct_data/ct_scans')
    data_list = []
    missing_files = []
    
    for idx, row in df.iterrows():
        basename = row['basename']
        folder = row['folder']
        
        # 构建文件路径
        image_path_template = config.get('image_path_template', '{base_path}/stanford_{folder}/{basename}.nii.gz')
        image_path = image_path_template.format(
            base_path=base_path,
            folder=folder,
            basename=basename
        )
        
        # 检查文件是否存在
        if config.get('check_file_exists', False):
            if not os.path.exists(image_path):
                missing_files.append(image_path)
                continue
        
        # 获取心脏功能指标数据
        cardiac_metrics = None
        if cardiac_metric_columns:
            try:
                cardiac_metrics = []
                for col in cardiac_metric_columns:
                    value = row[col]
                    if pd.isna(value):
                        cardiac_metrics = None
                        break
                    cardiac_metrics.append(float(value))
                
                if cardiac_metrics is not None:
                    cardiac_metrics = np.array(cardiac_metrics, dtype=np.float32)
            except (ValueError, TypeError) as e:
                print(f"警告: 行 {idx} 的心脏功能指标数据无效: {e}")
                cardiac_metrics = None
        
        # 收集其他元数据
        metadata = {}
        metadata_columns = config.get('metadata_columns', [])
        for col in metadata_columns:
            if col in row and pd.notna(row[col]):
                metadata[col] = row[col]
        
        data_item = {
            'image': image_path,
            'cardiac_metrics': cardiac_metrics,
            'patient_id': row.get('patient_id', basename),
            'basename': basename,
            'folder': folder,
            'metadata': metadata
        }
        
        data_list.append(data_item)
    
    print(f"成功构建 {len(data_list)} 个数据项")
    
    if missing_files and config.get('check_file_exists', False):
        print(f"警告: 有 {len(missing_files)} 个文件不存在")
        if len(missing_files) <= 5:
            print("缺失的文件:")
            for f in missing_files:
                print(f"  {f}")
        else:
            print(f"前5个缺失的文件:")
            for f in missing_files[:5]:
                print(f"  {f}")
    
    return data_list


def split_data(data_list, config):
    """分割数据为训练集和验证集"""
    split_method = config.get('split_method', 'random')
    split_ratio = config.get('train_val_split', 0.8)
    random_state = config.get('seed', 42)
    
    if split_method == 'random':
        # 随机分割
        train_data, val_data = train_test_split(
            data_list, 
            train_size=split_ratio, 
            random_state=random_state,
            shuffle=True
        )
    elif split_method == 'sequential':
        # 顺序分割
        split_idx = int(len(data_list) * split_ratio)
        train_data = data_list[:split_idx]
        val_data = data_list[split_idx:]
    elif split_method == 'patient_based':
        # 基于患者ID的分割（避免同一患者的数据出现在训练和验证集中）
        patient_ids = list(set([item['patient_id'] for item in data_list]))
        train_patients, val_patients = train_test_split(
            patient_ids,
            train_size=split_ratio,
            random_state=random_state,
            shuffle=True
        )
        
        train_data = [item for item in data_list if item['patient_id'] in train_patients]
        val_data = [item for item in data_list if item['patient_id'] in val_patients]
    else:
        raise ValueError(f"不支持的数据分割方法: {split_method}")
    
    print(f"数据分割完成:")
    print(f"  训练集: {len(train_data)} 个样本")
    print(f"  验证集: {len(val_data)} 个样本")
    
    return train_data, val_data


def create_data_loaders(config):
    """创建数据加载器"""
    # 加载和验证CSV数据
    df, cardiac_metric_columns = load_and_validate_csv_data(config)
    
    # 构建数据列表
    data_list = build_data_list(df, config, cardiac_metric_columns)
    
    if not data_list:
        raise ValueError("没有有效的数据项")
    
    # 分割数据
    train_data, val_data = split_data(data_list, config)
    
    # 创建数据集
    train_dataset = CardiacDataset(
        train_data, 
        cardiac_metric_columns=cardiac_metric_columns
    )
    
    val_dataset = None
    if val_data:
        val_dataset = CardiacDataset(
            val_data, 
            cardiac_metric_columns=cardiac_metric_columns
        )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        drop_last=True  # 训练时必须设置为True，避免BatchNorm错误
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.get('batch_size', 4),
            shuffle=False,
            num_workers=config.get('num_workers', 4),
            pin_memory=True,
            drop_last=False
        )
    
    # 保存数据统计信息
    data_info = {
        'total_samples': len(data_list),
        'train_samples': len(train_data),
        'val_samples': len(val_data) if val_data else 0,
        'cardiac_metric_columns': cardiac_metric_columns,
        'split_method': config.get('split_method', 'random'),
        'split_ratio': config.get('train_val_split', 0.8)
    }
    
    # 保存到输出目录
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'data_info.json', 'w', encoding='utf-8') as f:
        json.dump(data_info, f, indent=2, ensure_ascii=False)
    
    return train_loader, val_loader


 