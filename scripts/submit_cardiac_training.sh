#!/bin/bash
#SBATCH --job-name=cardiac_training
#SBATCH --partition=BMR-AI
#SBATCH --nodelist=stelvio
#SBATCH --gres=gpu:rtx8000:3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH --time=72:00:00
#SBATCH --output=logs/cardiac_training_%j.out
#SBATCH --error=logs/cardiac_training_%j.err

# 解析命令行参数
CONFIG_FILE=""
OUTPUT_DIR=""
EPOCHS=""
BATCH_SIZE=""
LEARNING_RATE=""
NUM_WORKERS=""

# 显示帮助信息
show_help() {
    echo "使用方法: sbatch $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --config FILE         配置文件路径 (默认: configs/fast_training_config.json)"
    echo "  --output_dir DIR      输出目录 (默认: 自动生成时间戳目录)"
    echo "  --epochs N            训练轮数"
    echo "  --batch_size N        批量大小"
    echo "  --learning_rate F     学习率"
    echo "  --num_workers N       数据加载器工作进程数"
    echo "  --help, -h            显示帮助信息"
    echo ""
    echo "示例:"
    echo "  sbatch $0 --config configs/fast_training_config.json --epochs 100"
    echo "  sbatch $0 --output_dir outputs/my_experiment --batch_size 32"
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --num_workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 设置默认值
CONFIG_FILE=${CONFIG_FILE:-"configs/fast_training_config.json"}

# 如果没有指定输出目录，自动生成唯一目录
if [ -z "$OUTPUT_DIR" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="outputs/fast_cardiac_training_${TIMESTAMP}_job${SLURM_JOB_ID}"
fi

# 创建必要的目录
mkdir -p logs
mkdir -p "$OUTPUT_DIR"

# 打印作业信息
echo "========================================"
echo "心脏功能训练作业信息"
echo "========================================"
echo "作业ID: $SLURM_JOB_ID"
echo "作业名称: $SLURM_JOB_NAME"
echo "节点: $SLURM_JOB_NODELIST"
echo "分区: $SLURM_JOB_PARTITION"
echo "GPU: $SLURM_JOB_GPUS"
echo "CPU核心: $SLURM_CPUS_PER_TASK"
echo "内存: $SLURM_MEM_PER_NODE MB"
echo "时间限制: $SLURM_TIME_LIMIT"
echo "========================================" 
echo "训练配置:"
echo "配置文件: $CONFIG_FILE"
echo "输出目录: $OUTPUT_DIR"
if [ -n "$EPOCHS" ]; then
    echo "训练轮数: $EPOCHS"
fi
if [ -n "$BATCH_SIZE" ]; then
    echo "批量大小: $BATCH_SIZE"
fi
if [ -n "$LEARNING_RATE" ]; then
    echo "学习率: $LEARNING_RATE"
fi
if [ -n "$NUM_WORKERS" ]; then
    echo "工作进程: $NUM_WORKERS"
fi
echo "========================================"

# 模块加载（根据您的集群环境调整）
echo "🔧 加载环境模块..."
# module load python/3.9
# module load cuda/11.8
# module load cudnn/8.6

# 激活Python环境（根据您的环境调整）
echo "🐍 激活Python环境..."
# source /path/to/your/conda/bin/activate your_env_name
# 或者
# source /path/to/your/venv/bin/activate

# 设置CUDA环境变量
export CUDA_VISIBLE_DEVICES=0,1,2
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# 设置Python路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 显示GPU信息
echo "🖥️  GPU信息:"
nvidia-smi

# 验证PyTorch GPU可用性
echo "🔍 验证PyTorch GPU..."
python3 -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}')"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 构建训练命令
echo "🚀 开始训练..."
TRAIN_CMD="python3 examples/cardiac_training_example.py --config $CONFIG_FILE --output_dir $OUTPUT_DIR"

# 添加可选参数
if [ -n "$EPOCHS" ]; then
    TRAIN_CMD="$TRAIN_CMD --epochs $EPOCHS"
fi
if [ -n "$BATCH_SIZE" ]; then
    TRAIN_CMD="$TRAIN_CMD --batch_size $BATCH_SIZE"
fi
if [ -n "$LEARNING_RATE" ]; then
    TRAIN_CMD="$TRAIN_CMD --learning_rate $LEARNING_RATE"
fi
if [ -n "$NUM_WORKERS" ]; then
    TRAIN_CMD="$TRAIN_CMD --num_workers $NUM_WORKERS"
fi

# 显示要执行的命令
echo "执行命令: $TRAIN_CMD"
echo "========================================"

# 执行训练
$TRAIN_CMD

# 获取退出状态
EXIT_CODE=$?

# 训练完成后的处理
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 训练成功完成！"
    echo "📁 输出文件位置: $OUTPUT_DIR"
    echo "📊 训练日志: $OUTPUT_DIR/training.log"
    echo "🏆 最佳模型: $OUTPUT_DIR/best_model.pth"
    echo "📈 TensorBoard: $OUTPUT_DIR/tensorboard"
    
    # 显示输出目录内容
    echo ""
    echo "📋 输出文件列表:"
    ls -la "$OUTPUT_DIR"
    
    # 显示最后几行训练日志
    if [ -f "$OUTPUT_DIR/training.log" ]; then
        echo ""
        echo "📝 训练日志最后10行:"
        tail -n 10 "$OUTPUT_DIR/training.log"
    fi
else
    echo "❌ 训练失败，退出码: $EXIT_CODE"
    echo "请检查错误日志: logs/cardiac_training_${SLURM_JOB_ID}.err"
fi

echo "========================================"
echo "作业完成时间: $(date)"
echo "作业ID: $SLURM_JOB_ID"
echo "输出目录: $OUTPUT_DIR"
echo "========================================"

exit $EXIT_CODE 