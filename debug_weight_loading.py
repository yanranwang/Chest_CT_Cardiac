#!/usr/bin/env python3
"""
调试权重加载问题 - 分析预训练权重结构
"""

import torch
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

def analyze_pretrained_weights():
    """分析预训练权重文件结构"""
    
    # 预训练权重文件路径
    pretrained_path = '/dataNAS/people/joycewyr/Chest_CT_Cardiac/merlin/models/checkpoints/i3_resnet_clinical_longformer_best_clip_04-02-2024_23-21-36_epoch_99.pt'
    
    print("🔍 分析预训练权重文件结构")
    print("=" * 80)
    
    try:
        # 加载权重文件
        print(f"📁 加载权重文件: {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location='cpu')
        
        print(f"✅ 权重文件加载成功，包含 {len(state_dict)} 个参数")
        
        # 分析权重键结构
        print("\n📊 权重键结构分析:")
        
        # 统计不同前缀的权重
        prefix_counts = {}
        encode_image_keys = []
        
        for key in state_dict.keys():
            if key.startswith('encode_image.'):
                encode_image_keys.append(key)
                
            # 提取前缀
            if '.' in key:
                prefix = key.split('.')[0]
                prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
        
        print(f"权重前缀统计:")
        for prefix, count in sorted(prefix_counts.items()):
            print(f"  {prefix}: {count} 个权重")
        
        # 分析 encode_image 相关的权重
        print(f"\n🎯 encode_image 相关权重 (共 {len(encode_image_keys)} 个):")
        
        # 按层分组显示
        layer_groups = {}
        for key in encode_image_keys:
            # 移除 encode_image. 前缀
            clean_key = key.replace('encode_image.', '')
            
            # 提取层名
            if clean_key.startswith('i3_resnet.'):
                layer_part = clean_key.replace('i3_resnet.', '')
                if '.' in layer_part:
                    layer_name = layer_part.split('.')[0]
                else:
                    layer_name = layer_part
                
                if layer_name not in layer_groups:
                    layer_groups[layer_name] = []
                layer_groups[layer_name].append(key)
        
        print("按层分组的权重:")
        for layer_name, keys in sorted(layer_groups.items()):
            print(f"  {layer_name}: {len(keys)} 个权重")
            if len(keys) <= 10:  # 如果权重少于10个，显示全部
                for key in keys:
                    print(f"    - {key}")
            else:  # 否则显示前5个和后5个
                for key in keys[:5]:
                    print(f"    - {key}")
                print(f"    ... ({len(keys) - 10} 个权重省略)")
                for key in keys[-5:]:
                    print(f"    - {key}")
        
        # 检查早期层权重
        print(f"\n🔍 早期层权重检查:")
        early_layers = ['conv1', 'bn1', 'layer1', 'layer2']
        
        for layer in early_layers:
            found_keys = [key for key in encode_image_keys if f'i3_resnet.{layer}' in key]
            if found_keys:
                print(f"  ✅ {layer}: 找到 {len(found_keys)} 个权重")
                for key in found_keys[:3]:  # 显示前3个
                    print(f"    - {key}")
                if len(found_keys) > 3:
                    print(f"    ... 还有 {len(found_keys) - 3} 个")
            else:
                print(f"  ❌ {layer}: 未找到权重")
        
        # 检查后期层权重
        print(f"\n🔍 后期层权重检查:")
        late_layers = ['layer3', 'layer4', 'classifier', 'contrastive_head']
        
        for layer in late_layers:
            found_keys = [key for key in encode_image_keys if f'i3_resnet.{layer}' in key]
            if found_keys:
                print(f"  ✅ {layer}: 找到 {len(found_keys)} 个权重")
                for key in found_keys[:3]:  # 显示前3个
                    print(f"    - {key}")
                if len(found_keys) > 3:
                    print(f"    ... 还有 {len(found_keys) - 3} 个")
            else:
                print(f"  ❌ {layer}: 未找到权重")
        
        # 分析权重映射
        print(f"\n🔧 权重映射分析:")
        
        # 模拟权重映射过程
        image_encoder_dict = {}
        for key, value in state_dict.items():
            if key.startswith('encode_image.i3_resnet.'):
                new_key = key.replace('encode_image.', '')
                image_encoder_dict[new_key] = value
        
        print(f"成功映射的权重数量: {len(image_encoder_dict)}")
        
        # 分析映射后的权重结构
        mapped_layer_groups = {}
        for key in image_encoder_dict.keys():
            layer_part = key.replace('i3_resnet.', '')
            if '.' in layer_part:
                layer_name = layer_part.split('.')[0]
            else:
                layer_name = layer_part
            
            if layer_name not in mapped_layer_groups:
                mapped_layer_groups[layer_name] = []
            mapped_layer_groups[layer_name].append(key)
        
        print("映射后的层权重分布:")
        for layer_name, keys in sorted(mapped_layer_groups.items()):
            print(f"  {layer_name}: {len(keys)} 个权重")
        
        # 检查所有键
        print(f"\n📋 所有权重键预览 (前20个):")
        for i, key in enumerate(sorted(state_dict.keys())[:20]):
            print(f"  {i+1:2d}. {key}")
        
        print(f"\n📋 所有权重键预览 (后20个):")
        for i, key in enumerate(sorted(state_dict.keys())[-20:]):
            print(f"  {len(state_dict) - 20 + i + 1:2d}. {key}")
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    analyze_pretrained_weights() 