import os
import shutil
from sklearn.model_selection import train_test_split

def prepare_data(config):
    """
    分别对 SAR–mask 与 RGB–mask 进行独立配对、划分并复制。
    每一对 (image, mask) 按 basename 精确匹配。
    """

    IMG_EXTS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')

    def list_files_with_ext(dirpath, exts):
        if not os.path.isdir(dirpath):
            return []
        return sorted([
            f for f in os.listdir(dirpath)
            if os.path.isfile(os.path.join(dirpath, f)) and os.path.splitext(f)[1].lower() in exts
        ])

    def match_pairs(img_dir, mask_dir, img_type):
        """按文件名匹配单一影像类型（SAR 或 RGB）与 mask"""
        img_files = list_files_with_ext(img_dir, IMG_EXTS)
        mask_files = list_files_with_ext(mask_dir, IMG_EXTS)

        print(f"[prepare_data] found {len(img_files)} {img_type} images in {img_dir}")
        print(f"[prepare_data] found {len(mask_files)} masks in {mask_dir}")

        img_map = {os.path.splitext(f)[0]: os.path.join(img_dir, f) for f in img_files}
        mask_map = {os.path.splitext(f)[0]: os.path.join(mask_dir, f) for f in mask_files}

        common = sorted(set(img_map.keys()) & set(mask_map.keys()))
        miss_mask = sorted(set(img_map.keys()) - set(mask_map.keys()))
        miss_img = sorted(set(mask_map.keys()) - set(img_map.keys()))

        print(f"[prepare_data] {img_type}–mask pairs: {len(common)}")
        if miss_mask:
            print(f"[prepare_data] WARNING: {len(miss_mask)} {img_type} images have no matching mask (examples): {miss_mask[:5]}")
        if miss_img:
            print(f"[prepare_data] WARNING: {len(miss_img)} masks have no matching {img_type} (examples): {miss_img[:5]}")

        imgs = [img_map[b] for b in common]
        masks = [mask_map[b] for b in common]
        return imgs, masks

    # --- 读取配置 ---
    sar_dir  = getattr(config, "SAR_IMAGE_DIR", None)
    rgb_dir  = getattr(config, "RGB_IMAGE_DIR", None)
    sar_mask_dir = getattr(config, "SAR_MASK_DIR", None)
    rgb_mask_dir = getattr(config, "RGB_MASK_DIR", None)
    sdg_dir      = getattr(config, "SDG_IMAGE_DIR", None)  # ✅ 新增SDG
    sdg_mask_dir = getattr(config, "SDG_MASK_DIR", None)
    data_root = getattr(config, "DATA_ROOT", "./dataset")

    train_ratio = getattr(config, "TRAIN_RATIO", 0.7)
    val_ratio   = getattr(config, "VAL_RATIO", 0.15)
    test_ratio  = getattr(config, "TEST_RATIO", 0.15)
    seed        = getattr(config, "SEED", 42)

    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total

    # --- 分别匹配 SAR–mask 与 RGB–mask ---
    all_results = {}

    modal_dirs = [
        ("SAR", sar_dir, sar_mask_dir),
        ("RGB", rgb_dir, rgb_mask_dir),
    ]

    for img_type, img_dir, mask_dir in modal_dirs:
        if img_dir is None or mask_dir is None or not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            print(f"[prepare_data] skip {img_type}: directory not found.")
            continue

        imgs, masks = match_pairs(img_dir, mask_dir, img_type)

        if len(imgs) == 0:
            print(f"[prepare_data] WARNING: no valid {img_type}–mask pairs found.")
            all_results[img_type] = ([], [], [])
            continue

        # --- 划分数据集 ---
        test_size = test_ratio
        if test_size > 0:
            imgs_trainval, imgs_test, masks_trainval, masks_test = train_test_split(
                imgs, masks, test_size=test_size, random_state=seed, shuffle=True
            )
        else:
            imgs_trainval, imgs_test, masks_trainval, masks_test = imgs, [], masks, []

        if val_ratio > 0:
            relative_val = val_ratio / (train_ratio + val_ratio)
            imgs_train, imgs_val, masks_train, masks_val = train_test_split(
                imgs_trainval, masks_trainval, test_size=relative_val, random_state=seed, shuffle=True
            )
        else:
            imgs_train, masks_train = imgs_trainval, masks_trainval
            imgs_val, masks_val = [], []

        print(f"[prepare_data] {img_type} -> train: {len(imgs_train)}, val: {len(imgs_val)}, test: {len(imgs_test)}")

        # --- 保存 ---
        subsets = {
            'train': (imgs_train, masks_train),
            'val':   (imgs_val, masks_val),
            'test':  (imgs_test, masks_test),
        }

        for subset_name, (subset_imgs, subset_masks) in subsets.items():
            img_out = os.path.join(data_root, img_type, subset_name, "images")
            mask_out = os.path.join(data_root, img_type, subset_name, "segmentation_masks")
            os.makedirs(img_out, exist_ok=True)
            os.makedirs(mask_out, exist_ok=True)

            for src_img, src_mask in zip(subset_imgs, subset_masks):
                shutil.copy2(src_img, os.path.join(img_out, os.path.basename(src_img)))
                shutil.copy2(src_mask, os.path.join(mask_out, os.path.basename(src_mask)))

        all_results[img_type] = (
            (imgs_train, masks_train),
            (imgs_val, masks_val),
            (imgs_test, masks_test)
        )

    print(f"[prepare_data] ✅ All done. Output root: {data_root}")
    return all_results
