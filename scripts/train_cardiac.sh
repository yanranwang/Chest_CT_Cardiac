#!/bin/bash
"""
心脏功能训练脚本
==================

这个脚本提供了多种训练模式和配置选项，用于快速启动心脏功能回归训练。

使用方法:
  ./train_cardiac.sh [模式] [选项]

支持的模式:
  basic     - 基础训练
  fast      - 快速训练 (使用预处理数据)
  debug     - 调试模式 (小批量)
  production - 生产模式 (完整训练)
  resume    - 恢复训练
  custom    - 自定义配置

示例:
  ./train_cardiac.sh basic
  ./train_cardiac.sh fast --epochs 50
  ./train_cardiac.sh debug --batch_size 2
  ./train_cardiac.sh custom --config my_config.json
"""

# 设置默认参数
DEFAULT_EPOCHS=30
DEFAULT_BATCH_SIZE=32
DEFAULT_LEARNING_RATE=1e-4
DEFAULT_OUTPUT_DIR="outputs/cardiac_training"
DEFAULT_CSV_PATH="/dataNAS/people/joycewyr/Chest_CT_Cardiac/filtered_echo_chestCT_data_filtered_chest_data.csv"
DEFAULT_DEVICE="cuda"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印函数
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_header() {
    echo -e "${BLUE}"
    echo "========================================"
    echo "$1"
    echo "========================================"
    echo -e "${NC}"
}

# 显示帮助信息
show_help() {
    echo "心脏功能训练脚本"
    echo ""
    echo "用法: $0 [模式] [选项]"
    echo ""
    echo "模式:"
    echo "  basic       基础训练模式"
    echo "  fast        快速训练模式 (使用预处理数据)"
    echo "  debug       调试模式 (小批量, 少epochs)"
    echo "  production  生产模式 (完整训练配置)"
    echo "  resume      恢复训练模式"
    echo "  custom      自定义配置模式"
    echo ""
    echo "选项:"
    echo "  --config FILE        配置文件路径"
    echo "  --epochs N           训练轮数 (默认: $DEFAULT_EPOCHS)"
    echo "  --batch_size N       批量大小 (默认: $DEFAULT_BATCH_SIZE)"
    echo "  --learning_rate F    学习率 (默认: $DEFAULT_LEARNING_RATE)"
    echo "  --output_dir DIR     输出目录 (默认: $DEFAULT_OUTPUT_DIR)"
    echo "  --csv_path FILE      CSV数据文件路径"
    echo "  --device DEVICE      训练设备 (默认: $DEFAULT_DEVICE)"
    echo "  --resume_from FILE   恢复训练的检查点文件"
    echo "  --use_fast_loader    使用快速数据加载器"
    echo "  --preprocessed_dir DIR 预处理数据目录"
    echo "  --num_workers N      数据加载器worker数量"
    echo "  --dry_run            仅显示命令，不执行"
    echo "  --help, -h           显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 basic"
    echo "  $0 fast --epochs 50"
    echo "  $0 debug --batch_size 2"
    echo "  $0 production --output_dir /path/to/output"
    echo "  $0 custom --config my_config.json"
    echo "  $0 resume --resume_from outputs/checkpoint.pth"
}

# 检查依赖
check_dependencies() {
    print_info "检查依赖..."
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 未安装"
        return 1
    fi
    
    # 检查训练脚本
    if [ ! -f "examples/cardiac_training_example.py" ]; then
        print_error "训练脚本不存在: examples/cardiac_training_example.py"
        return 1
    fi
    
    print_success "依赖检查通过"
    return 0
}

# 创建输出目录
create_output_dir() {
    local output_dir="$1"
    if [ ! -d "$output_dir" ]; then
        print_info "创建输出目录: $output_dir"
        mkdir -p "$output_dir"
    fi
}

# 基础训练模式
run_basic_training() {
    print_header "基础训练模式"
    
    local cmd="python3 examples/cardiac_training_example.py"
    cmd="$cmd --epochs ${EPOCHS:-$DEFAULT_EPOCHS}"
    cmd="$cmd --batch_size ${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}"
    cmd="$cmd --learning_rate ${LEARNING_RATE:-$DEFAULT_LEARNING_RATE}"
    cmd="$cmd --output_dir ${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
    cmd="$cmd --device ${DEVICE:-$DEFAULT_DEVICE}"
    
    if [ -n "$CSV_PATH" ]; then
        cmd="$cmd --csv_path $CSV_PATH"
    fi
    
    if [ -n "$NUM_WORKERS" ]; then
        cmd="$cmd --num_workers $NUM_WORKERS"
    fi
    
    execute_command "$cmd"
}

# 快速训练模式
run_fast_training() {
    print_header "快速训练模式"
    
    # 检查预处理数据
    local preprocessed_dir="${PREPROCESSED_DIR:-/data/joycewyr/cardiac_training_fast}"
    if [ ! -d "$preprocessed_dir" ]; then
        print_warning "预处理数据目录不存在: $preprocessed_dir"
        print_info "请先运行数据预处理:"
        print_info "python3 -m merlin.training.data_preprocessor --config config.json"
        return 1
    fi
    
    local cmd="python3 examples/cardiac_training_example.py"
    cmd="$cmd --use_fast_loader"
    cmd="$cmd --preprocessed_data_dir $preprocessed_dir"
    cmd="$cmd --epochs ${EPOCHS:-$DEFAULT_EPOCHS}"
    cmd="$cmd --batch_size ${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}"
    cmd="$cmd --learning_rate ${LEARNING_RATE:-$DEFAULT_LEARNING_RATE}"
    cmd="$cmd --output_dir ${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
    cmd="$cmd --device ${DEVICE:-$DEFAULT_DEVICE}"
    
    if [ -n "$NUM_WORKERS" ]; then
        cmd="$cmd --num_workers $NUM_WORKERS"
    fi
    
    execute_command "$cmd"
}

# 调试模式
run_debug_training() {
    print_header "调试训练模式"
    
    local cmd="python3 examples/cardiac_training_example.py"
    cmd="$cmd --epochs ${EPOCHS:-10}"
    cmd="$cmd --batch_size ${BATCH_SIZE:-2}"
    cmd="$cmd --learning_rate ${LEARNING_RATE:-$DEFAULT_LEARNING_RATE}"
    cmd="$cmd --output_dir ${OUTPUT_DIR:-outputs/debug_training}"
    cmd="$cmd --device ${DEVICE:-$DEFAULT_DEVICE}"
    cmd="$cmd --log_interval 1"
    cmd="$cmd --save_interval 5"
    
    if [ -n "$NUM_WORKERS" ]; then
        cmd="$cmd --num_workers $NUM_WORKERS"
    fi
    
    execute_command "$cmd"
}

# 生产模式
run_production_training() {
    print_header "生产训练模式"
    
    local cmd="python3 examples/cardiac_training_example.py"
    cmd="$cmd --epochs ${EPOCHS:-200}"
    cmd="$cmd --batch_size ${BATCH_SIZE:-8}"
    cmd="$cmd --learning_rate ${LEARNING_RATE:-2e-4}"
    cmd="$cmd --output_dir ${OUTPUT_DIR:-outputs/production_training}"
    cmd="$cmd --device ${DEVICE:-$DEFAULT_DEVICE}"
    cmd="$cmd --log_interval 10"
    cmd="$cmd --save_interval 10"
    
    if [ -n "$CSV_PATH" ]; then
        cmd="$cmd --csv_path $CSV_PATH"
    fi
    
    if [ -n "$NUM_WORKERS" ]; then
        cmd="$cmd --num_workers $NUM_WORKERS"
    fi
    
    execute_command "$cmd"
}

# 恢复训练模式
run_resume_training() {
    print_header "恢复训练模式"
    
    if [ -z "$RESUME_FROM" ]; then
        print_error "请指定恢复检查点文件: --resume_from FILE"
        return 1
    fi
    
    if [ ! -f "$RESUME_FROM" ]; then
        print_error "检查点文件不存在: $RESUME_FROM"
        return 1
    fi
    
    local cmd="python3 examples/cardiac_training_example.py"
    cmd="$cmd --resume_from $RESUME_FROM"
    cmd="$cmd --epochs ${EPOCHS:-$DEFAULT_EPOCHS}"
    cmd="$cmd --batch_size ${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}"
    cmd="$cmd --learning_rate ${LEARNING_RATE:-$DEFAULT_LEARNING_RATE}"
    cmd="$cmd --output_dir ${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
    cmd="$cmd --device ${DEVICE:-$DEFAULT_DEVICE}"
    
    if [ -n "$NUM_WORKERS" ]; then
        cmd="$cmd --num_workers $NUM_WORKERS"
    fi
    
    execute_command "$cmd"
}

# 自定义配置模式
run_custom_training() {
    print_header "自定义配置模式"
    
    if [ -z "$CONFIG_FILE" ]; then
        print_error "请指定配置文件: --config FILE"
        return 1
    fi
    
    if [ ! -f "$CONFIG_FILE" ]; then
        print_error "配置文件不存在: $CONFIG_FILE"
        return 1
    fi
    
    local cmd="python3 examples/cardiac_training_example.py"
    cmd="$cmd --config $CONFIG_FILE"
    
    # 如果有额外参数，添加到命令中
    if [ -n "$EPOCHS" ]; then cmd="$cmd --epochs $EPOCHS"; fi
    if [ -n "$BATCH_SIZE" ]; then cmd="$cmd --batch_size $BATCH_SIZE"; fi
    if [ -n "$LEARNING_RATE" ]; then cmd="$cmd --learning_rate $LEARNING_RATE"; fi
    if [ -n "$OUTPUT_DIR" ]; then cmd="$cmd --output_dir $OUTPUT_DIR"; fi
    if [ -n "$DEVICE" ]; then cmd="$cmd --device $DEVICE"; fi
    if [ -n "$NUM_WORKERS" ]; then cmd="$cmd --num_workers $NUM_WORKERS"; fi
    
    execute_command "$cmd"
}

# 执行命令
execute_command() {
    local cmd="$1"
    
    print_info "执行命令:"
    echo "  $cmd"
    echo ""
    
    if [ "$DRY_RUN" = "true" ]; then
        print_warning "干运行模式 - 命令未执行"
        return 0
    fi
    
    # 创建输出目录
    create_output_dir "${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
    
    # 执行命令
    eval "$cmd"
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        print_success "训练完成"
    else
        print_error "训练失败，退出码: $exit_code"
    fi
    
    return $exit_code
}

# 解析参数
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --config)
                CONFIG_FILE="$2"
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
            --output_dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --csv_path)
                CSV_PATH="$2"
                shift 2
                ;;
            --device)
                DEVICE="$2"
                shift 2
                ;;
            --resume_from)
                RESUME_FROM="$2"
                shift 2
                ;;
            --use_fast_loader)
                USE_FAST_LOADER="true"
                shift
                ;;
            --preprocessed_dir)
                PREPROCESSED_DIR="$2"
                shift 2
                ;;
            --num_workers)
                NUM_WORKERS="$2"
                shift 2
                ;;
            --dry_run)
                DRY_RUN="true"
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                if [ -z "$MODE" ]; then
                    MODE="$1"
                else
                    print_error "未知参数: $1"
                    show_help
                    exit 1
                fi
                shift
                ;;
        esac
    done
}

# 主函数
main() {
    # 解析参数
    parse_args "$@"
    
    # 默认模式
    if [ -z "$MODE" ]; then
        MODE="basic"
    fi
    
    # 检查依赖
    if ! check_dependencies; then
        exit 1
    fi
    
    # 执行对应模式
    case "$MODE" in
        basic)
            run_basic_training
            ;;
        fast)
            run_fast_training
            ;;
        debug)
            run_debug_training
            ;;
        production)
            run_production_training
            ;;
        resume)
            run_resume_training
            ;;
        custom)
            run_custom_training
            ;;
        *)
            print_error "未知模式: $MODE"
            show_help
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@" 