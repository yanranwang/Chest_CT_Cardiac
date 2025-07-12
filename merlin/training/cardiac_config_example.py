"""
心脏功能回归训练配置示例
支持从CSV文件读取数据进行训练
"""

def get_training_config():
    """获取训练配置"""
    config = {
        # 基本训练配置
        'output_dir': 'outputs/cardiac_training',
        'pretrained_model_path': None,  # 设置为Merlin预训练模型路径
        'num_cardiac_metrics': 10,
        'epochs': 100,
        'batch_size': 4,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'optimizer': 'adamw',
        'loss_function': 'mse',
        'scheduler': {
            'type': 'cosine',
            'eta_min': 1e-6
        },
        'freeze_encoder': False,
        'grad_clip': 1.0,
        'device': 'cuda',
        'seed': 42,
        'num_workers': 4,
        'log_interval': 10,
        'save_interval': 10,
        'use_tensorboard': True,
        'drop_last': False,
        
        # 数据分割配置
        'train_val_split': 0.8,
        'split_method': 'random',  # 选项: 'random', 'sequential', 'patient_based'
        
        # CSV数据配置
        'csv_path': 'filtered_echo_chestCT_data_filtered_chest_data.csv',
        'required_columns': ['basename', 'folder'],
        
        # 心脏功能指标列（如果CSV中包含真实的心脏功能数据）
        'cardiac_metric_columns': [
            # 示例列名 - 请根据实际CSV文件修改
            # 'ejection_fraction',
            # 'end_diastolic_volume',
            # 'end_systolic_volume',
            # 'stroke_volume',
            # 'cardiac_output',
            # 'left_ventricular_mass',
            # 'wall_thickness',
            # 'wall_motion_score',
            # 'diastolic_function',
            # 'systolic_function'
        ],
        
        # 额外的元数据列
        'metadata_columns': ['patient_id', 'age', 'gender', 'diagnosis'],
        
        # 文件路径配置
        'base_path': '/dataNAS/data/ct_data/ct_scans',
        'image_path_template': '{base_path}/stanford_{folder}/{basename}.nii.gz',
        'check_file_exists': False,  # 设置为True以在训练前检查所有文件是否存在
        
        # 数据清理配置
        'remove_missing_files': True,   # 移除缺失basename或folder的行
        'remove_duplicates': True,      # 移除重复数据
    }
    
    return config


def get_demo_config():
    """获取演示配置（使用模拟数据）"""
    config = get_training_config()
    
    # 演示配置修改
    config.update({
        'epochs': 5,
        'batch_size': 2,
        'check_file_exists': False,
        'cardiac_metric_columns': [],  # 使用模拟数据
        'train_val_split': 0.7,
        'split_method': 'random'
    })
    
    return config


def get_production_config():
    """获取生产环境配置"""
    config = get_training_config()
    
    # 生产环境配置
    config.update({
        'epochs': 200,
        'batch_size': 8,
        'learning_rate': 5e-5,
        'check_file_exists': True,
        'split_method': 'patient_based',  # 避免数据泄露
        'grad_clip': 0.5,
        'scheduler': {
            'type': 'plateau',
            'factor': 0.5,
            'patience': 15
        },
        # 设置真实的心脏功能指标列
        'cardiac_metric_columns': [
            'ejection_fraction',
            'end_diastolic_volume', 
            'end_systolic_volume',
            'stroke_volume',
            'cardiac_output'
        ]
    })
    
    return config


if __name__ == '__main__':
    """运行训练脚本的示例"""
    from cardiac_trainer import create_data_loaders, CardiacTrainer
    
    # 选择配置
    config = get_demo_config()  # 可以改为 get_production_config()
    
    print("配置信息:")
    print(f"CSV文件路径: {config['csv_path']}")
    print(f"数据分割方法: {config['split_method']}")
    print(f"训练/验证比例: {config['train_val_split']}")
    print(f"心脏功能指标列: {config['cardiac_metric_columns']}")
    print(f"检查文件存在: {config['check_file_exists']}")
    print()
    
    try:
        # 创建数据加载器
        train_loader, val_loader = create_data_loaders(config)
        
        # 创建训练器
        trainer = CardiacTrainer(config)
        
        # 开始训练
        trainer.train(train_loader, val_loader)
        
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        print("请检查:")
        print("1. CSV文件路径是否正确")
        print("2. CSV文件中是否包含必需的列")
        print("3. 图像文件路径是否正确")
        print("4. 心脏功能指标列名是否正确")
        raise 