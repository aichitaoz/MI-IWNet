import rasterio

tif_path = "/home/xiaobowen/project/internal_wave_detection_project/IW_data/images_sar/S1A_IW_GRDH_1SDV_20250814T211813_20250814T211838_060537_078747_C3A3.SAFE.tiff"

with rasterio.open(tif_path) as src:
    print("========== BASIC INFO ==========")
    print("Driver:", src.driver)
    print("Width:", src.width)
    print("Height:", src.height)
    print("Count (bands):", src.count)
    print("Dtype:", src.dtypes)
    print("CRS:", src.crs)

    print("\n========== TRANSFORM ==========")
    print(src.transform)

    print("\n========== BOUNDS ==========")
    print("left   (min_lon):", src.bounds.left)
    print("right  (max_lon):", src.bounds.right)
    print("bottom (min_lat):", src.bounds.bottom)
    print("top    (max_lat):", src.bounds.top)

    print("\n========== RESOLUTION ==========")
    print("x_res:", src.res[0])
    print("y_res:", src.res[1])

    print("\n========== TAGS (METADATA) ==========")
    for k, v in src.tags().items():
        print(f"{k}: {v}")
