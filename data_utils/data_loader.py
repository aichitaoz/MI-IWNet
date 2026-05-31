
import os
import shutil
from sklearn.model_selection import train_test_split

def prepare_data(config):
    """准备数据集，进行训练、验证、测试划分，并保存到对应目录结构中"""

    # 获取所有图像和掩码文件，按文件名排序
    image_files = sorted([f for f in os.listdir(config.IMAGE_DIR) if f.endswith('.jpg')])
    mask_files = sorted([f for f in os.listdir(config.MASK_DIR) if f.endswith('.jpg')])
    
    # 拼接成完整路径
    image_paths = [os.path.join(config.IMAGE_DIR, f) for f in image_files]
    mask_paths = [os.path.join(config.MASK_DIR, f) for f in mask_files]

    print(f"Found {len(image_paths)} image-mask pairs")
    
    # 训练集和临时集划分
    train_imgs, temp_imgs, train_masks, temp_masks = train_test_split(
        image_paths, mask_paths, test_size=(1 - config.TRAIN_RATIO), random_state=config.SEED
    )
    
    # 验证集和测试集划分（从 temp 中继续分）
    val_imgs, test_imgs, val_masks, test_masks = train_test_split(
        temp_imgs, temp_masks,
        test_size=config.TEST_RATIO / (config.VAL_RATIO + config.TEST_RATIO),
        random_state=config.SEED
    )

    print(f"Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")

    # 构建目标路径结构
    subsets = {
        'train': (train_imgs, train_masks),
        'val': (val_imgs, val_masks),
        'test': (test_imgs, test_masks),
    }

    for subset_name, (imgs, masks) in subsets.items():
        img_dir = os.path.join(config.DATA_ROOT, subset_name, "images")
        mask_dir = os.path.join(config.DATA_ROOT, subset_name, "segmentation_masks")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        print(f"Saving {subset_name} set to {img_dir} and {mask_dir} ...")

        for img_path, mask_path in zip(imgs, masks):
            shutil.copy(img_path, os.path.join(img_dir, os.path.basename(img_path)))
            shutil.copy(mask_path, os.path.join(mask_dir, os.path.basename(mask_path)))

    return (train_imgs, train_masks), (val_imgs, val_masks), (test_imgs, test_masks)