import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================== 环境修复 ==============================
import os
import warnings

# 1. 临时清空 LD_LIBRARY_PATH，避免系统 CUDA/cuDNN 干扰
os.environ['LD_LIBRARY_PATH'] = ''

# 2. 可选：禁用 CUDA 的一些 legacy 搜索路径
if 'CUDA_HOME' in os.environ:
    del os.environ['CUDA_HOME']

# 3. 警告提示
warnings.warn(
    "已清空 LD_LIBRARY_PATH，使 PyTorch 自带 CUDA/cuDNN 生效。"
    "如果仍报错，请确认 PyTorch 版本匹配 CUDA。"
)

# 4. 验证 GPU 是否可用
import torch
print(f"PyTorch CUDA version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print(f"CUDA available: {torch.cuda.is_available()}")

from configs.train_config import Config
from utils.set_seed import set_seed
from train.train_model import train_model
from train.logger import Logger

if __name__ == "__main__":
    Config = Config()
    set_seed(Config.SEED)

    # 日志
    log_file_path = os.path.join(Config.LOG_SAVE_DIR, 'log', f"{Config.MODEL_TYPE}_train_log.txt")
    sys.stdout = Logger(log_file_path)

    print(f"Training {Config.MODEL_TYPE} on device: {Config.DEVICE}")

    # 训练（train_model内部已包含完整的训练、验证和测试流程）
    model = train_model(Config, feature_save_dir=Config.FEATURE_SAVE_DIR)

    print("\n" + "="*80)
    print("✅ Training completed successfully!")
    print("="*80)
    print(f"Best model saved to: {Config.PTH_SAVE_DIR}/best_{Config.MODEL_TYPE}_model.pth")
    print(f"Training logs saved to: {log_file_path}")
    print(f"Training curves saved to: {Config.FIG_SAVE_DIR}")
