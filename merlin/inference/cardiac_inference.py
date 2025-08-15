import os
import torch
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Union, Optional

from merlin.models.cardiac_regression import CardiacFunctionModel, CardiacMetricsCalculator
from merlin.data.monai_transforms import ImageTransforms


class CardiacInference:
    """心脏功能推理类"""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Initialize心脏功能推理器
        
        Args:
            model_path: 训练好的模型权重路径
            device: 设备 ('auto', 'cuda', 'cpu')
        """
        self.device = self._setup_device(device)
        self.model = self._load_model(model_path)
        self.transform = ImageTransforms
        self.metric_names = CardiacMetricsCalculator.get_metric_names()
        
        # Setlog
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"心脏功能推理器Initialize完成，使用设备: {self.device}")
    
    def _setup_device(self, device: str) -> torch.device:
        """设置Calculate设备"""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.device(device)
    
    def _load_model(self, model_path: str) -> CardiacFunctionModel:
        """Load训练好的模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型file does not exist: {model_path}")
        
        # Createmodelinstance
        model = CardiacFunctionModel()
        
        # Loadweights
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # ProcessDataParallel包装的model
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # RemoveDataParallel前缀
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
        
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        
        self.logger.info(f"成功Load模型: {model_path}")
        return model
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        预ProcessCT图像
        
        Args:
            image_path: CT图像文件路径
            
        Returns:
            preprocessed_image: 预Process后的图像张量
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像file does not exist: {image_path}")
        
        # Builddata字典
        data = {'image': image_path}
        
        # TODO: Translate '应用预'Processtransform
        data = self.transform(data)
        
        # Addbatchdimension并移动to设备
        image = data['image'].unsqueeze(0).to(self.device)
        
        return image
    
    def predict_single(self, image_path: str, normalize: bool = True) -> Dict[str, float]:
        """
        预测单个CT图像的心脏功能
        
        Args:
            image_path: CT图像文件路径
            normalize: 是否将预测值标准化到生理Range
            
        Returns:
            predictions: 心脏功能预测结果字典
        """
        # TODO: Translate '预'Processimage
        image = self.preprocess_image(image_path)
        
        # model推理
        with torch.no_grad():
            cardiac_predictions = self.model.predict_cardiac_function(image)
        
        # standard化Predictvalue
        if normalize:
            cardiac_predictions = CardiacMetricsCalculator.normalize_predictions(cardiac_predictions)
        
        # Convert为numpy并移回CPU
        cardiac_predictions = cardiac_predictions.cpu().numpy().flatten()
        
        # Buildresults字典
        results = {}
        for i, metric_name in enumerate(self.metric_names):
            results[metric_name] = float(cardiac_predictions[i])
        
        return results
    
    def predict_batch(self, image_paths: List[str], normalize: bool = True) -> List[Dict[str, float]]:
        """
        批量预测多个CT图像的心脏功能
        
        Args:
            image_paths: CT图像文件路径列表
            normalize: 是否将预测值标准化到生理Range
            
        Returns:
            predictions_list: 心脏功能预测结果列表
        """
        all_results = []
        
        for image_path in image_paths:
            try:
                result = self.predict_single(image_path, normalize)
                result['image_path'] = image_path
                result['status'] = 'success'
                all_results.append(result)
                
                self.logger.info(f"成功预测: {image_path}")
                
            except Exception as e:
                error_result = {
                    'image_path': image_path,
                    'status': 'error',
                    'error_message': str(e)
                }
                all_results.append(error_result)
                
                self.logger.error(f"预测失败 {image_path}: {str(e)}")
        
        return all_results
    
    def predict_with_confidence(self, image_path: str, num_samples: int = 10) -> Dict[str, Dict[str, float]]:
        """
        使用蒙特卡洛dropout预测心脏功能，提供不Determine性估计
        
        Args:
            image_path: CT图像文件路径
            num_samples: 蒙特卡洛采样次数
            
        Returns:
            predictions_with_uncertainty: 包含均值、Standard deviation的预测结果
        """
        # enabledropout进行notDetermine性估计
        self.model.train()
        
        # TODO: Translate '预'Processimage
        image = self.preprocess_image(image_path)
        
        # TODO: Translate '多次采样'all_predictions = []
        with torch.no_grad():
            for _ in range(num_samples):
                cardiac_predictions = self.model.predict_cardiac_function(image)
                cardiac_predictions = CardiacMetricsCalculator.normalize_predictions(cardiac_predictions)
                all_predictions.append(cardiac_predictions.cpu().numpy().flatten())
        
        # restoreevalmode
        self.model.eval()
        
        # Calculatestatistics量
        all_predictions = np.array(all_predictions)  # [num_samples, num_metrics]
        
        results = {}
        for i, metric_name in enumerate(self.metric_names):
            results[metric_name] = {
                'mean': float(np.mean(all_predictions[:, i])),
                'std': float(np.std(all_predictions[:, i])),
                'min': float(np.min(all_predictions[:, i])),
                'max': float(np.max(all_predictions[:, i])),
                'confidence_interval_95': [
                    float(np.percentile(all_predictions[:, i], 2.5)),
                    float(np.percentile(all_predictions[:, i], 97.5))
                ]
            }
        
        return results
    
    def generate_report(self, predictions: Dict[str, float], patient_id: str = None) -> str:
        """
        生成心脏功能预测报告
        
        Args:
            predictions: 心脏功能预测结果
            patient_id: 患者ID
            
        Returns:
            report: 格式化的报告文本
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("心脏功能预测报告")
        report_lines.append("=" * 60)
        
        if patient_id:
            report_lines.append(f"患者ID: {patient_id}")
        
        report_lines.append(f"预测时间: {torch.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # cardiac function指标
        report_lines.append("心脏功能指标:")
        report_lines.append("-" * 40)
        
        metric_descriptions = {
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
        
        for metric_name, value in predictions.items():
            if metric_name in metric_descriptions:
                description = metric_descriptions[metric_name]
                report_lines.append(f"{description:25}: {value:8.2f}")
        
        report_lines.append("")
        report_lines.append("-" * 40)
        report_lines.append("注意：本预测结果仅供临床参考，不能替代专业医学诊断")
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)
    
    def save_predictions(self, predictions: Union[Dict, List[Dict]], output_path: str):
        """
        Save预测结果到文件
        
        Args:
            predictions: 预测结果
            output_path: 输出文件路径
        """
        import json
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add元data
        if isinstance(predictions, dict):
            output_data = {
                'metadata': {
                    'model_type': 'cardiac_function_regression',
                    'metric_names': self.metric_names,
                    'prediction_time': torch.datetime.now().isoformat()
                },
                'predictions': predictions
            }
        else:
            output_data = {
                'metadata': {
                    'model_type': 'cardiac_function_regression',
                    'metric_names': self.metric_names,
                    'prediction_time': torch.datetime.now().isoformat(),
                    'num_samples': len(predictions)
                },
                'predictions': predictions
            }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"预测结果已Save到: {output_path}")


def main():
    """演示心脏功能推理"""
    # configparameters
    model_path = "outputs/cardiac_training/best_model.pth"  # Train好的modelpath
    image_path = "path/to/ct_scan.nii.gz"  # CTimagepath
    
    try:
        # Create推理器
        predictor = CardiacInference(model_path)
        
        # singlePredict
        predictions = predictor.predict_single(image_path)
        print("心脏功能预测结果:")
        for metric, value in predictions.items():
            print(f"  {metric}: {value:.2f}")
        
        # Generatereport
        report = predictor.generate_report(predictions, patient_id="PATIENT_001")
        print("\n" + report)
        
        # notDetermine性估计
        predictions_with_uncertainty = predictor.predict_with_confidence(image_path)
        print("\n不Determine性估计:")
        for metric, stats in predictions_with_uncertainty.items():
            print(f"  {metric}: {stats['mean']:.2f} ± {stats['std']:.2f}")
        
        # Saveresults
        predictor.save_predictions(predictions, "outputs/predictions.json")
        
    except Exception as e:
        print(f"推理失败: {str(e)}")


if __name__ == "__main__":
    main() 