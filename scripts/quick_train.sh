#!/bin/bash

# 快速心脏训练启动脚本
# 提供几个常用的训练命令选项

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🚀 心脏功能训练快速启动${NC}"
echo "=================================="

# 显示选项
echo "请选择训练模式:"
echo "1. 基础训练 (100 epochs, batch_size=4)"
echo "2. 快速训练 (使用预处理数据)"
echo "3. 调试模式 (10 epochs, batch_size=2)"
echo "4. 生产模式 (200 epochs, batch_size=8)"
echo "5. 自定义配置"
echo "6. 恢复训练"
echo "0. 退出"
echo ""

read -p "请输入选择 (0-6): " choice

case $choice in
    1)
        echo -e "${GREEN}启动基础训练...${NC}"
        ./scripts/train_cardiac.sh basic
        ;;
    2)
        echo -e "${GREEN}启动快速训练...${NC}"
        ./scripts/train_cardiac.sh fast
        ;;
    3)
        echo -e "${GREEN}启动调试模式...${NC}"
        ./scripts/train_cardiac.sh debug
        ;;
    4)
        echo -e "${GREEN}启动生产模式...${NC}"
        ./scripts/train_cardiac.sh production
        ;;
    5)
        read -p "请输入配置文件路径: " config_path
        if [ -f "$config_path" ]; then
            echo -e "${GREEN}使用自定义配置启动训练...${NC}"
            ./scripts/train_cardiac.sh custom --config "$config_path"
        else
            echo "配置文件不存在: $config_path"
        fi
        ;;
    6)
        read -p "请输入检查点文件路径: " checkpoint_path
        if [ -f "$checkpoint_path" ]; then
            echo -e "${GREEN}恢复训练...${NC}"
            ./scripts/train_cardiac.sh resume --resume_from "$checkpoint_path"
        else
            echo "检查点文件不存在: $checkpoint_path"
        fi
        ;;
    0)
        echo "退出"
        exit 0
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac 