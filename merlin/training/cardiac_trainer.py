import os
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import monai
from monai.losses import DiceLoss
from monai.transforms import Compose
import logging
from pathlib import Path
from tqdm import tqdm
import warnings

# TODO: Translate '抑制'not必要的warning
warnings.filterwarnings('ignore', category=UserWarning)

from merlin.models.cardiac_regression import CardiacFunctionModel, CardiacMetricsCalculator
from merlin.data.dataloaders import CTPersistentDataset
from merlin.data.monai_transforms import ImageTransforms


class CardiacDataset(Dataset):
    """心脏功能数据集 - 支持从CSV文件Read数据"""
    def __init__(self, data_list, transform=None, cardiac_metric_columns=None):
        self.data_list = data_list
        self.transform = transform or ImageTransforms
        self.cardiac_metric_columns = cardiac_metric_columns or []
        self.metric_names = CardiacMetricsCalculator.get_metric_names()
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        sample = self.data_list[idx]
        
        # Loadand预Processimage
        if self.transform:
            sample = self.transform(sample)
        
        # Getcardiac functionlabels
        if 'cardiac_metrics' in sample and sample['cardiac_metrics'] is not None:
            cardiac_metrics = torch.tensor(sample['cardiac_metrics'], dtype=torch.float32)
        else:
            # if没has真实labels，抛出error而notis使用模拟data
            raise ValueError(f"Sample {idx} (patient_id: {sample.get('patient_id', 'unknown')}) lacks cardiac function label data。"
                           f"请确保CSV文件中包含有效的心脏功能指标列，或Check数据预Process过程。")
        
        # TODO: Translate '确保'labelsincludeLVEFandAS两个value
        if len(cardiac_metrics) < 2:
            # iflabelscountnot足，抛出error而notis使用模拟data
            raise ValueError(f"Sample {idx} (patient_id: {sample.get('patient_id', 'unknown')}) 的心脏功能标签数量不足。"
                           f"期望至少2个标签(LVEF和AS)，但只有 {len(cardiac_metrics)} 个。"
                           f"请Checkcardiac_metric_columns配置和CSV数据。")
        
        return {
            'image': sample['image'],
            'cardiac_metrics': cardiac_metrics,
            'patient_id': sample.get('patient_id', f'patient_{idx}'),
            'basename': sample.get('basename', f'unknown_{idx}'),
            'folder': sample.get('folder', 'unknown'),
            'image_path': sample.get('image', ''),
            'metadata': sample.get('metadata', {})
        }
    
class CardiacTrainer:
    """Cardiac function regression trainer"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Create output directory
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging (must be before using logger)
        self._setup_logging()
        
        # Set random seed
        self._set_random_seed(config.get('seed', 42))
        
        # Initialize model
        self.model = self._build_model()
        
        # Initialize optimizer and scheduler
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # Initialize loss function
        self.criterion = self._build_loss_function()
        
        # Initialize training records
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.epoch_times = []
        
        # Initialize tensorboard writer
        if config.get('use_tensorboard', True) and TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(self.output_dir / 'tensorboard')
        else:
            self.writer = None
            if config.get('use_tensorboard', True) and not TENSORBOARD_AVAILABLE:
                self.logger.warning("TensorBoard不可用，跳过TensorBoard日志记录")
    
    def _set_random_seed(self, seed):
        """Set random seed"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    
    def _build_model(self):
        """Build model"""
        model = CardiacFunctionModel(
            pretrained_model_path=self.config.get('pretrained_model_path')
        )
        
        # Whether to freeze image encoder
        if self.config.get('freeze_encoder', False):
            model.freeze_encoder(True)
            self.logger.info("Image encoder frozen")
        
        model = model.to(self.device)
        
        # Multi-GPU training
        if torch.cuda.device_count() > 1:
            # Check if batch_size is large enough to avoid batch_size=1 per GPU
            batch_size = self.config.get('batch_size', 4)
            num_gpus = torch.cuda.device_count()
            if batch_size < num_gpus:
                self.logger.warning(f"Batch size {batch_size} is smaller than GPU count {num_gpus}, may cause BatchNorm error")
                self.logger.warning("Consider increasing batch_size or reducing GPU count")
            
            model = nn.DataParallel(model)
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs for training")
        
        return model
    
    def _build_optimizer(self):
        """Build optimizer"""
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
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        return optimizer
    
    def _build_scheduler(self):
        """Build learning rate scheduler"""
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
        """Build loss function"""
        from merlin.models.cardiac_regression import CardiacLoss
        
        # Use specialized cardiac function loss that handles LVEF regression and AS classification
        regression_weight = self.config.get('regression_weight', 1.0)
        classification_weight = self.config.get('classification_weight', 1.0)
        
        # Calculateclass别weights（ifenable）
        class_weights = None
        if self.config.get('use_class_weights', False):
            class_weights = self._calculate_class_weights()
            if class_weights is not None:
                self.logger.info(f"使用Class权重: {class_weights.tolist()}")
        
        criterion = CardiacLoss(
            regression_weight=regression_weight,
            classification_weight=classification_weight,
            class_weights=class_weights
        )
        
        return criterion
    
    def _calculate_class_weights(self):
        """CalculateAS分类的Class权重"""
        try:
            if hasattr(self, 'train_loader') and self.train_loader is not None:
                # fromTraindataLoad器中statisticslabels分布
                as_labels = []
                for batch in self.train_loader:
                    if 'as_maybe' in batch:
                        as_labels.extend(batch['as_maybe'].cpu().numpy())
                    elif 'AS_maybe' in batch:
                        as_labels.extend(batch['AS_maybe'].cpu().numpy())
                    elif 'labels' in batch and len(batch['labels'].shape) > 1:
                        # TODO: Translate '假设'ASlabelsissecond列
                        as_labels.extend(batch['labels'][:, 1].cpu().numpy())
                
                if len(as_labels) > 0:
                    as_labels = np.array(as_labels)
                    unique, counts = np.unique(as_labels, return_counts=True)
                    total = len(as_labels)
                    
                    # Calculateweights：total / (class别数 * 每classsample数)
                    weights = total / (len(unique) * counts)
                    
                    # Createweights张量，index对应class别labels
                    class_weights = torch.zeros(2)  # TODO: Translate '假设只'has0and1两class
                    for i, label in enumerate(unique):
                        class_weights[int(label)] = weights[i]
                    
                    return class_weights
            
            return None
        except Exception as e:
            self.logger.warning(f"CalculateClass权重失败: {e}")
            return None
    
    def _print_label_distribution(self):
        """Print训练和Validation set的标签分布"""
        try:
            self.logger.info("=" * 60)
            self.logger.info("📊 数据集Label Distribution Statistics")
            self.logger.info("=" * 60)
            
            # statisticsTrain集
            if hasattr(self, 'train_loader') and self.train_loader is not None:
                train_stats = self._get_dataset_stats(self.train_loader, "Training set")
                
            # statisticsvalidate集
            if hasattr(self, 'val_loader') and self.val_loader is not None:
                val_stats = self._get_dataset_stats(self.val_loader, "Validation set")
                
            self.logger.info("=" * 60)
            
        except Exception as e:
            self.logger.warning(f"Statistics标签分布失败: {e}")
    
    def _get_dataset_stats(self, dataloader, dataset_name):
        """Get数据集Statisticsinfo"""
        lvef_values = []
        as_labels = []
        
        # TODO: Translate '临时'Set为Evaluatemode以避免影响Train
        original_training = self.model.training
        self.model.eval()
        
        try:
            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    # TODO: Translate '只'statistics前几个batch以加fast度
                    if i >= 10:  # TODO: Translate '最多'statistics10个batch
                        break
                        
                    if 'lvef' in batch:
                        lvef_values.extend(batch['lvef'].cpu().numpy())
                    if 'as_maybe' in batch:
                        as_labels.extend(batch['as_maybe'].cpu().numpy())
                    elif 'AS_maybe' in batch:
                        as_labels.extend(batch['AS_maybe'].cpu().numpy())
                    elif 'labels' in batch and len(batch['labels'].shape) > 1:
                        lvef_values.extend(batch['labels'][:, 0].cpu().numpy())
                        as_labels.extend(batch['labels'][:, 1].cpu().numpy())
        finally:
            # restore原始Trainmode
            self.model.train(original_training)
        
        # statisticsLVEF
        if len(lvef_values) > 0:
            lvef_values = np.array(lvef_values)
            self.logger.info(f"{dataset_name} LVEFStatistics:")
            self.logger.info(f"  Sample数: {len(lvef_values)}")
            self.logger.info(f"  均值: {lvef_values.mean():.2f}")
            self.logger.info(f"  Standard deviation: {lvef_values.std():.2f}")
            self.logger.info(f"  Range: [{lvef_values.min():.2f}, {lvef_values.max():.2f}]")
        
        # statisticsASlabels
        if len(as_labels) > 0:
            as_labels = np.array(as_labels)
            unique, counts = np.unique(as_labels, return_counts=True)
            total = len(as_labels)
            
            self.logger.info(f"{dataset_name} AS分类Statistics:")
            self.logger.info(f"  总Sample数: {total}")
            for label, count in zip(unique, counts):
                percentage = (count / total) * 100
                self.logger.info(f"  Class {int(label)}: {count} Sample ({percentage:.1f}%)")
            
            # Calculate正负sample比例
            if len(unique) == 2:
                pos_count = counts[unique == 1][0] if 1 in unique else 0
                neg_count = counts[unique == 0][0] if 0 in unique else 0
                if neg_count > 0:
                    ratio = pos_count / neg_count
                    self.logger.info(f"  正负Sample比例: 1:{ratio:.2f}")
        
        return {'lvef_stats': lvef_values if len(lvef_values) > 0 else None,
                'as_stats': as_labels if len(as_labels) > 0 else None}
    
    def _setup_logging(self):
        """Setup logging"""
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
        """Format time display"""
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
        """Train one epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(train_loader)
        
        # Create progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1:3d}/{self.config.get("epochs", 100)}', 
                   ncols=120, leave=False)
        
        batch_losses = []
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            
            # Extract LVEF and AS target values from cardiac_metrics
            cardiac_metrics = batch['cardiac_metrics'].to(self.device)
            # Assume first value is LVEF, second is AS
            if cardiac_metrics.shape[1] >= 2:
                lvef_targets = cardiac_metrics[:, 0]  # LVEF target
                as_targets = cardiac_metrics[:, 1]    # AS target
            else:
                # If only one value, assume it's LVEF, set AS to 0
                lvef_targets = cardiac_metrics[:, 0]
                as_targets = torch.zeros_like(lvef_targets)
            
            # Forward pass
            self.optimizer.zero_grad()
            lvef_preds, as_preds = self.model(images)
            
            # Calculate loss
            loss_dict = self.criterion(lvef_preds, as_preds, lvef_targets, as_targets)
            loss = loss_dict['total_loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            
            self.optimizer.step()
            
            # Record loss
            batch_losses.append(loss.item())
            epoch_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Log training progress
            if batch_idx % self.config.get('log_interval', 10) == 0:
                self.logger.info(
                    f'Epoch {epoch+1:3d}/{self.config.get("epochs", 100)} '
                    f'[{batch_idx:4d}/{num_batches:4d}] '
                    f'Loss: {loss.item():.6f} '
                    f'LR: {self.optimizer.param_groups[0]["lr"]:.2e}'
                )
        
        pbar.close()
        
        # Calculate average loss
        avg_loss = epoch_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate_epoch(self, val_loader, epoch):
        """Validate一个epoch"""
        self.model.eval()
        epoch_loss = 0.0
        all_predictions = []
        all_targets = []
        
        # Createvalidate进度条
        pbar = tqdm(val_loader, desc=f'Validating', ncols=120, leave=False)
        
        with torch.no_grad():
            for batch in pbar:
                images = batch['image'].to(self.device)
                
                # fromcardiac_metrics中ExtractLVEFandAS目标value
                cardiac_metrics = batch['cardiac_metrics'].to(self.device)
                if cardiac_metrics.shape[1] >= 2:
                    lvef_targets = cardiac_metrics[:, 0]  # LVEF目标value
                    as_targets = cardiac_metrics[:, 1]    # AS目标value
                else:
                    lvef_targets = cardiac_metrics[:, 0]
                    as_targets = torch.zeros_like(lvef_targets)
                
                # TODO: Translate '前向传播'lvef_preds, as_preds = self.model(images)
                
                # Calculateloss
                loss_dict = self.criterion(lvef_preds, as_preds, lvef_targets, as_targets)
                loss = loss_dict['total_loss']
                epoch_loss += loss.item()
                
                # updatevalidate进度条
                pbar.set_postfix({'Val Loss': f'{loss.item():.4f}'})
                
                # TODO: Translate '收集'Predictand真value用于指标Calculate
                predictions = torch.stack([lvef_preds.squeeze(), as_preds.squeeze()], dim=1)
                targets = torch.stack([lvef_targets, as_targets], dim=1)
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        pbar.close()
        
        avg_loss = epoch_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        
        # CalculateEvaluate指标
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        metrics = self._calculate_metrics(all_predictions, all_targets)
        
        # recordvalidateresults
        self.logger.info(f'Validate结果 - Epoch {epoch+1:3d} | Loss: {avg_loss:.6f}')
        for metric_name, value in metrics.items():
            self.logger.info(f'  {metric_name}: {value:.4f}')
        
        if self.writer:
            self.writer.add_scalar('Val/Loss', avg_loss, epoch)
            for metric_name, value in metrics.items():
                self.writer.add_scalar(f'Val/{metric_name}', value, epoch)
        
        return avg_loss, metrics
    
    def _calculate_metrics(self, predictions, targets):
        """Calculate评估指标"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {}
        
        # LVEFregression指标（第0列）
        lvef_preds = predictions[:, 0]
        lvef_targets = targets[:, 0]
        
        metrics['LVEF_MSE'] = mean_squared_error(lvef_targets, lvef_preds)
        metrics['LVEF_MAE'] = mean_absolute_error(lvef_targets, lvef_preds)
        metrics['LVEF_R2'] = r2_score(lvef_targets, lvef_preds)
        
        # ASclassification指标（第1列）
        as_preds = predictions[:, 1]
        as_targets = targets[:, 1]
        
        # TODO: Translate '将概率'Convert为二classificationPredict
        as_pred_binary = (as_preds > 0.5).astype(int)
        as_targets_binary = as_targets.astype(int)
        
        metrics['AS_Accuracy'] = accuracy_score(as_targets_binary, as_pred_binary)
        metrics['AS_Precision'] = precision_score(as_targets_binary, as_pred_binary, zero_division=0)
        metrics['AS_Recall'] = recall_score(as_targets_binary, as_pred_binary, zero_division=0)
        metrics['AS_F1'] = f1_score(as_targets_binary, as_pred_binary, zero_division=0)
        
        # CalculateAS的AUC
        try:
            from sklearn.metrics import roc_auc_score
            metrics['AS_AUC'] = roc_auc_score(as_targets_binary, as_preds)
        except ValueError:
            metrics['AS_AUC'] = 0.0
        
        return metrics
    
    def save_checkpoint(self, epoch, is_best=False):
        """SaveCheck点"""
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
        
        # Save最新Checkpoint
        torch.save(checkpoint, self.output_dir / 'checkpoint_latest.pth')
        
        # Save最佳model
        if is_best:
            torch.save(checkpoint, self.output_dir / 'checkpoint_best.pth')
            torch.save(self.model.state_dict(), self.output_dir / 'best_model.pth')
            self.logger.info(f'💾 Save最佳模型 (Epoch {epoch+1}, Val Loss: {self.best_val_loss:.6f})')
    
    def load_checkpoint(self, checkpoint_path):
        """LoadCheck点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        self.logger.info(f'从 {checkpoint_path} LoadCheck点')
        return checkpoint['epoch']
    
    def train(self, train_loader, val_loader=None, start_epoch=0):
        """完整训练流程"""
        epochs = self.config.get('epochs', 100)
        
        # TODO: Translate '存储'dataLoad器供其他method使用
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # PrintTrainstartinfo
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
        print(f"📁 数据Statistics:")
        print(f"   Training set大小: {len(train_loader.dataset)}")
        if val_loader:
            print(f"   Validation set大小: {len(val_loader.dataset)}")
        print(f"   每轮批次数: {len(train_loader)}")
        print(f"💾 输出目录: {self.output_dir}")
        if self.writer:
            print(f"📈 TensorBoard: {self.output_dir / 'tensorboard'}")
        print("=" * 80)
        
        self.logger.info(f'开始训练，共 {epochs} 个epoch')
        
        # Printlabels分布statistics
        self._print_label_distribution()
        
        # Trainstarttime
        training_start_time = time.time()
        
        for epoch in range(start_epoch, epochs):
            epoch_start_time = time.time()
            
            # Printepochstartinfo
            print(f"\n🔄 Epoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # validate
            val_loss, val_metrics = None, {}
            if val_loader:
                val_loss, val_metrics = self.validate_epoch(val_loader, epoch)
                
                # updatelearning rate调度器
                if self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                
                # Checkis否为最佳model
                is_best = val_loss < self.best_val_loss
                if is_best:
                    improvement = self.best_val_loss - val_loss
                    self.best_val_loss = val_loss
                    print(f"✨ 新的最佳模型！Validate损失降低了 {improvement:.6f}")
                
                # SaveCheckpoint
                if epoch % self.config.get('save_interval', 10) == 0 or is_best:
                    self.save_checkpoint(epoch, is_best)
            else:
                # TODO: Translate '无'validate集时
                if self.scheduler and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
                
                if epoch % self.config.get('save_interval', 10) == 0:
                    self.save_checkpoint(epoch)
            
            # Calculatetime
            epoch_time = time.time() - epoch_start_time
            self.epoch_times.append(epoch_time)
            
            # TODO: Translate '预估剩余'time
            avg_epoch_time = np.mean(self.epoch_times[-10:])  # TODO: Translate '使用最近10个'epoch的averagetime
            remaining_epochs = epochs - epoch - 1
            estimated_remaining_time = avg_epoch_time * remaining_epochs
            
            # Printepoch总结
            print(f"📊 Epoch {epoch+1} 总结:")
            print(f"   训练损失: {train_loss:.6f}")
            if val_loss is not None:
                print(f"   Validate损失: {val_loss:.6f}")
                print(f"   LVEF R²: {val_metrics.get('LVEF_R2', 0):.4f}")
                print(f"   AS 准确率: {val_metrics.get('AS_Accuracy', 0):.4f}")
            print(f"   轮次耗时: {self._format_time(epoch_time)}")
            print(f"   平均耗时: {self._format_time(avg_epoch_time)}")
            print(f"   预估剩余: {self._format_time(estimated_remaining_time)}")
            print(f"   当前学习率: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # recordtotensorboard
            if self.writer:
                self.writer.add_scalar('Train/EpochLoss', train_loss, epoch)
                self.writer.add_scalar('Train/EpochTime', epoch_time, epoch)
                if val_loader:
                    self.writer.add_scalar('Val/EpochLoss', val_loss, epoch)
        
        # Traincomplete
        total_training_time = time.time() - training_start_time
        
        print("\n" + "=" * 80)
        print("🎉 训练完成！")
        print("=" * 80)
        print(f"📈 训练Statistics:")
        print(f"   总训练时间: {self._format_time(total_training_time)}")
        print(f"   平均每轮时间: {self._format_time(np.mean(self.epoch_times))}")
        print(f"   最佳Validate损失: {self.best_val_loss:.6f}")
        print(f"💾 输出文件:")
        print(f"   最佳模型: {self.output_dir}/best_model.pth")
        print(f"   训练日志: {self.output_dir}/training.log")
        print(f"   配置文件: {self.output_dir}/config.json")
        if self.writer:
            print(f"   TensorBoard: {self.output_dir}/tensorboard")
        print("=" * 80)
        
        self.logger.info('训练完成！')
        
        # Save最终model
        self.save_checkpoint(epochs - 1)
        
        # SaveTrainconfig
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        if self.writer:
            self.writer.close()


def load_and_validate_csv_data(config):
    """Load和ValidateCSV数据"""
    csv_path = config.get('csv_path')
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")
    
    print(f"从 {csv_path} Read数据...")
    df = pd.read_csv(csv_path)
    print(f"原始数据集大小: {len(df)} 行")
    
    # Check必需的列
    required_columns = config.get('required_columns', ['basename', 'folder'])
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"CSV文件中Missing required columns: {missing_columns}")
    
    # Checkcardiac function指标列
    cardiac_metric_columns = config.get('cardiac_metric_columns', [])
    if not cardiac_metric_columns:
        raise ValueError("配置中必须指定cardiac_metric_columns，不能使用模拟数据进行训练。"
                        "请在配置文件中设置cardiac_metric_columns，例如: ['lvef', 'as_maybe']")
    
    missing_cardiac_columns = [col for col in cardiac_metric_columns if col not in df.columns]
    if missing_cardiac_columns:
        raise ValueError(f"CSV文件中缺少必需的心脏功能指标列: {missing_cardiac_columns}。"
                        f"可用的列: {list(df.columns)}")
    
    print(f"找到心脏功能指标列: {cardiac_metric_columns}")
    
    # Checkis否至少需要2个cardiac function指标（LVEFandAS）
    if len(cardiac_metric_columns) < 2:
        raise ValueError(f"至少需要2个心脏功能指标列（LVEF和AS），但只提供了 {len(cardiac_metric_columns)} 个: {cardiac_metric_columns}")
    
    # dataClean
    if config.get('remove_missing_files', True):
        initial_count = len(df)
        df = df.dropna(subset=['basename', 'folder'])
        df = df[df['basename'].notna() & df['folder'].notna()]
        if len(df) < initial_count:
            print(f"Remove了 {initial_count - len(df)} 行缺失basename或folder的数据")
    
    # Remove重复项
    if config.get('remove_duplicates', True):
        initial_count = len(df)
        df = df.drop_duplicates(subset=['basename', 'folder'])
        if len(df) < initial_count:
            print(f"Remove了 {initial_count - len(df)} 行重复数据")
    
    print(f"Clean后数据集大小: {len(df)} 行")
    return df, cardiac_metric_columns


def build_data_list(df, config, cardiac_metric_columns):
    """从DataFrameBuild数据列表"""
    base_path = config.get('base_path', '/dataNAS/data/ct_data/ct_scans')
    data_list = []
    missing_files = []
    
    for idx, row in df.iterrows():
        basename = row['basename']
        folder = row['folder']
        
        # Buildfilepath
        image_path_template = config.get('image_path_template', '{base_path}/stanford_{folder}/{basename}.nii.gz')
        image_path = image_path_template.format(
            base_path=base_path,
            folder=folder,
            basename=basename
        )
        
        # Checkfileis否存in
        if config.get('check_file_exists', False):
            if not os.path.exists(image_path):
                missing_files.append(image_path)
                continue
        
        # Getcardiac function指标data
        cardiac_metrics = None
        try:
            cardiac_metrics = []
            for col in cardiac_metric_columns:
                value = row[col]
                if pd.isna(value):
                    print(f"警告: 行 {idx} (basename: {basename}) 的列 '{col}' 缺少数据，跳过该Sample")
                    cardiac_metrics = None
                    break
                cardiac_metrics.append(float(value))
            
            if cardiac_metrics is not None:
                cardiac_metrics = np.array(cardiac_metrics, dtype=np.float32)
        except (ValueError, TypeError) as e:
            print(f"警告: 行 {idx} (basename: {basename}) 的心脏功能指标数据无效: {e}，跳过该Sample")
            cardiac_metrics = None
        
        # if没hashas效的cardiac function指标data，skip这个sample
        if cardiac_metrics is None:
            continue
        
        # TODO: Translate '收集其他元'data
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
    
    print(f"成功Build {len(data_list)} 个数据项")
    
    if missing_files and config.get('check_file_exists', False):
        print(f"警告: 有 {len(missing_files)} 个file does not exist")
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
    """Split数据为Training set和Validation set"""
    split_method = config.get('split_method', 'random')
    split_ratio = config.get('train_val_split', 0.8)
    random_state = config.get('seed', 42)
    
    if split_method == 'random':
        # randomSplit
        train_data, val_data = train_test_split(
            data_list, 
            train_size=split_ratio, 
            random_state=random_state,
            shuffle=True
        )
    elif split_method == 'sequential':
        # sequenceSplit
        split_idx = int(len(data_list) * split_ratio)
        train_data = data_list[:split_idx]
        val_data = data_list[split_idx:]
    elif split_method == 'patient_based':
        # TODO: Translate '基于患者'ID的Split（避免同一患者的data出现inTrainandvalidate集中）
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
        raise ValueError(f"不支持的数据Split method: {split_method}")
    
    print(f"数据Split完成:")
    print(f"  Training set: {len(train_data)} samples")
    print(f"  Validation set: {len(val_data)} samples")
    
    return train_data, val_data


def create_data_loaders(config):
    """Create数据Load器"""
    # LoadandvalidateCSVdata
    df, cardiac_metric_columns = load_and_validate_csv_data(config)
    
    # Builddata列表
    data_list = build_data_list(df, config, cardiac_metric_columns)
    
    if not data_list:
        raise ValueError("没有有效的数据项")
    
    # Splitdata
    train_data, val_data = split_data(data_list, config)
    
    # Createdata集
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
    
    # CreatedataLoad器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 4),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True,
        drop_last=True  # Train时必须Set为True，避免BatchNormerror
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
    
    # Savedatastatisticsinfo
    data_info = {
        'total_samples': len(data_list),
        'train_samples': len(train_data),
        'val_samples': len(val_data) if val_data else 0,
        'cardiac_metric_columns': cardiac_metric_columns,
        'split_method': config.get('split_method', 'random'),
        'split_ratio': config.get('train_val_split', 0.8)
    }
    
    # SavetoOutput目录
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'data_info.json', 'w', encoding='utf-8') as f:
        json.dump(data_info, f, indent=2, ensure_ascii=False)
    
    return train_loader, val_loader

