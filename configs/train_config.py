import os
import torch

class Config:
    # ========================= 数据路径 =========================
    DATA_ROOT = '/home/xiaobowen/project/internal_wave_detection_project/IW_data'
    
    # 各模态输入路径
    SAR_IMAGE_DIR = os.path.join(DATA_ROOT, 'images_sar')
    RGB_IMAGE_DIR = os.path.join(DATA_ROOT, 'images_rgb')
    SAR_MASK_DIR = os.path.join(DATA_ROOT, 'masks_sar')
    RGB_MASK_DIR = os.path.join(DATA_ROOT, 'masks_rgb')
    

    
    # ========================= 模型与训练参数 =========================
    IMG_SIZE = 1024
    BATCH_SIZE = 1       # ✅ 方案A：交替训练，每次只处理1张图（SAR/RGB）
    NUM_EPOCHS = 100

    LEARNING_RATE = 2e-4  # 从3e-4提高到1e-3，帮助模型更快收敛
    WEIGHT_DECAY = 1e-4

    # ✅ 方案A：使用梯度累积保持等效batch_size
    # ✅ 新策略：SAR:RGB = 1:2 交替训练
    # 实际效果：每6步更新一次（2个SAR + 4个RGB），等效batch_size=6
    # 顺序：SAR, RGB, RGB, SAR, RGB, RGB, ... (然后更新参数)
    ACCUMULATION_STEPS = 6  # ✅ 配合1:2比例（2+4=6）

    # 模型类型（可选：unet, unet_fusion）
    MODEL_TYPE = 'ConvNeXt'  # ✅ 使用带原型融合的模型
    COMPARE_MODELS = False
    model_types = ['unet','ConvNeXt','UNetPlusPlus','SegFormer','TransUNet','IWResNet','SwinUNet','IWENet','mtu2net']

    # ========================= 原型融合参数 =========================
    USE_PROTOTYPE_FUSION = True  # 是否启用原型融合
    NUM_PROTOTYPES = [32, 64, 128, 256]  # 各层原型数量
    PROTOTYPE_LOSS_WEIGHT = 0.1  # picop原型损失权重
    CONVNEXT_ARCH = 'femto'  # ConvNeXt架构

    # ========================= Domain Adaptation参数 =========================
    # 每个stage的DA层reduction参数 [stage0, stage1, stage2, stage3]
    # ✅ 新策略：只在stage 0,1,2之后添加DA层（而非每个block）
    #
    # 设计理念：
    # - 浅层(stage0/1): 保留更多低级特征，用较小的reduction (8)
    # - 中层(stage2): 需要更强的跨模态对齐，用较大的reduction (16)
    # - 深层(stage3): 语义已经很抽象，不再需要DA
    #
    # SimplifiedDALayer特点：
    # - 参数量减少70% (相比GatedDALayer)
    # - 使用FiLM条件归一化 (Feature-wise Linear Modulation)
    # - 所有模态共享变换网络，用模态嵌入作为条件
    DA_REDUCTIONS = [8, 8, 16]  # ⚠️ 注意：只有3个值，对应stage 0,1,2

    # 注意：
    # - reduction越小，DA层表达能力越强，但参数量也越大
    # - 典型范围: 8~32，推荐 [8, 8, 16]
    # - 如果过拟合严重，可以增大reduction (如 [16, 16, 32])

    # ========================= 数据划分比例 =========================
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.2
    TEST_RATIO = 0.1

    # ========================= 设备与保存路径 =========================
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    LOG_SAVE_DIR = 'checkpoints'
    PTH_SAVE_DIR = 'checkpoints/revise/pth_rgb'
    FIG_SAVE_DIR = 'checkpoints/fig'

    FEATURE_SAVE_DIR = os.path.join(DATA_ROOT, 'features')

    # ========================= 其他配置 =========================
    POS_WEIGHT = 2.0
    SEED = 42
