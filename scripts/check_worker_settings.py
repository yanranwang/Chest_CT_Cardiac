#!/usr/bin/env python3
"""
Worker数量优化检查脚本
====================

这个脚本分析当前系统配置，并提供不同训练场景下的worker数量建议。
"""

import os
import json
import torch
from pathlib import Path


def get_system_info():
    """获取系统信息"""
    info = {
        'cpu_count': os.cpu_count(),
        'cuda_available': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    # 尝试获取内存信息
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
            for line in meminfo.split('\n'):
                if 'MemTotal:' in line:
                    mem_kb = int(line.split()[1])
                    info['memory_gb'] = mem_kb / (1024**2)
                elif 'MemAvailable:' in line:
                    mem_kb = int(line.split()[1])
                    info['available_memory_gb'] = mem_kb / (1024**2)
    except:
        info['memory_gb'] = 0
        info['available_memory_gb'] = 0
    
    if info['cuda_available']:
        info['gpu_memory_gb'] = []
        for i in range(info['gpu_count']):
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            info['gpu_memory_gb'].append(gpu_mem)
    
    return info


def calculate_worker_recommendations(system_info):
    """计算不同场景的worker数量建议"""
    cpu_count = system_info['cpu_count']
    memory_gb = system_info['memory_gb']
    
    recommendations = {
        'debug': {
            'num_workers': 0,
            'description': '调试模式 - 避免多进程问题',
            'use_case': '代码调试、错误排查'
        },
        'development': {
            'num_workers': min(4, cpu_count),
            'description': '开发模式 - 平衡性能和稳定性',
            'use_case': '开发测试、快速迭代'
        },
        'training': {
            'num_workers': min(8, cpu_count // 2),
            'description': '训练模式 - 平衡数据加载和计算',
            'use_case': '标准训练任务'
        },
        'production': {
            'num_workers': min(16, cpu_count // 4),
            'description': '生产模式 - 最大化训练效率',
            'use_case': '正式训练、性能优化'
        },
        'high_performance': {
            'num_workers': min(32, cpu_count // 2),
            'description': '高性能模式 - 最大化数据加载速度',
            'use_case': '大数据集、高GPU利用率'
        }
    }
    
    # 根据内存调整建议
    if memory_gb < 32:
        for mode in recommendations:
            recommendations[mode]['num_workers'] = min(recommendations[mode]['num_workers'], 4)
            recommendations[mode]['description'] += ' (内存限制)'
    
    return recommendations


def analyze_current_config():
    """分析当前配置文件的worker设置"""
    config_files = [
        'examples/cardiac_training_example.py',
        'configs/cardiac_config.json',
        'merlin/training/cardiac_config_example.py'
    ]
    
    current_settings = {}
    
    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                if config_file.endswith('.json'):
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                        current_settings[config_file] = config.get('num_workers', 'Not set')
                else:
                    # 从Python文件中提取worker设置
                    with open(config_file, 'r') as f:
                        content = f.read()
                        if "'num_workers'" in content:
                            lines = content.split('\n')
                            for line in lines:
                                if "'num_workers'" in line and ':' in line:
                                    worker_value = line.split(':')[1].strip().rstrip(',')
                                    current_settings[config_file] = worker_value
                                    break
            except Exception as e:
                current_settings[config_file] = f'Error reading: {e}'
    
    return current_settings


def print_analysis():
    """打印完整的分析报告"""
    print("=" * 80)
    print("🔍 Worker数量优化分析报告")
    print("=" * 80)
    
    # 系统信息
    system_info = get_system_info()
    print("\n📊 系统信息:")
    print(f"   CPU核心数: {system_info['cpu_count']}")
    print(f"   总内存: {system_info['memory_gb']:.1f} GB")
    print(f"   可用内存: {system_info['available_memory_gb']:.1f} GB")
    print(f"   CUDA可用: {'✅' if system_info['cuda_available'] else '❌'}")
    if system_info['cuda_available']:
        print(f"   GPU数量: {system_info['gpu_count']}")
        for i, gpu_mem in enumerate(system_info['gpu_memory_gb']):
            print(f"   GPU {i} 内存: {gpu_mem:.1f} GB")
    
    # 当前配置
    print("\n📋 当前配置:")
    current_settings = analyze_current_config()
    for config_file, worker_setting in current_settings.items():
        print(f"   {config_file}: {worker_setting}")
    
    # 推荐设置
    print("\n💡 推荐设置:")
    recommendations = calculate_worker_recommendations(system_info)
    
    for mode, config in recommendations.items():
        print(f"\n   🎯 {mode.upper()}模式:")
        print(f"      num_workers: {config['num_workers']}")
        print(f"      描述: {config['description']}")
        print(f"      适用场景: {config['use_case']}")
    
    # 性能影响分析
    print("\n📈 性能影响分析:")
    print("   num_workers = 0:")
    print("      ✅ 优点: 避免多进程问题，内存使用最少")
    print("      ❌ 缺点: 数据加载可能成为瓶颈，GPU利用率低")
    print("   num_workers = 4-8:")
    print("      ✅ 优点: 平衡性能和稳定性，适合大多数场景")
    print("      ❌ 缺点: 可能未充分利用系统资源")
    print("   num_workers = 16+:")
    print("      ✅ 优点: 最大化数据加载速度，提高GPU利用率")
    print("      ❌ 缺点: 内存使用较高，可能出现多进程问题")
    
    # 具体建议
    print("\n🎯 具体建议:")
    if system_info['cpu_count'] >= 16:
        print("   1. 开发阶段: 使用 num_workers=4-8")
        print("   2. 训练阶段: 使用 num_workers=8-16")
        print("   3. 生产阶段: 使用 num_workers=16-32")
    else:
        print("   1. 开发阶段: 使用 num_workers=2-4")
        print("   2. 训练阶段: 使用 num_workers=4-8")
        print("   3. 生产阶段: 使用 num_workers=8-16")
    
    print("\n⚠️  注意事项:")
    print("   - 从较小的worker数量开始，逐步增加")
    print("   - 监控GPU利用率，确保数据加载不是瓶颈")
    print("   - 注意内存使用，避免OOM错误")
    print("   - 在多GPU训练时，可能需要更多worker")
    
    print("\n🔧 快速设置命令:")
    print("   # 开发模式")
    print(f"   ./scripts/train_cardiac.sh debug --num_workers {recommendations['development']['num_workers']}")
    print("   # 训练模式")
    print(f"   ./scripts/train_cardiac.sh basic --num_workers {recommendations['training']['num_workers']}")
    print("   # 生产模式")
    print(f"   ./scripts/train_cardiac.sh production --num_workers {recommendations['production']['num_workers']}")
    
    print("=" * 80)


def update_config_file(config_path, num_workers):
    """更新配置文件中的worker数量"""
    try:
        if config_path.endswith('.json'):
            with open(config_path, 'r') as f:
                config = json.load(f)
            config['num_workers'] = num_workers
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"✅ 已更新 {config_path}: num_workers = {num_workers}")
        else:
            print(f"⚠️  Python文件 {config_path} 需要手动更新")
    except Exception as e:
        print(f"❌ 更新配置文件失败: {e}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Worker数量优化检查')
    parser.add_argument('--update-config', type=str, help='更新指定配置文件的worker数量')
    parser.add_argument('--num-workers', type=int, help='设置worker数量')
    parser.add_argument('--mode', type=str, choices=['debug', 'development', 'training', 'production', 'high_performance'], 
                       help='根据模式自动设置worker数量')
    
    args = parser.parse_args()
    
    if args.update_config and args.num_workers is not None:
        update_config_file(args.update_config, args.num_workers)
    elif args.update_config and args.mode:
        system_info = get_system_info()
        recommendations = calculate_worker_recommendations(system_info)
        recommended_workers = recommendations[args.mode]['num_workers']
        update_config_file(args.update_config, recommended_workers)
    else:
        print_analysis()


if __name__ == '__main__':
    main() 