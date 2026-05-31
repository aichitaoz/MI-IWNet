import os
import re
import requests
import rasterio

TIFF_FOLDER = "./sars"
EMAIL = "aichitaozi4814@gmail.com"
PASSWORD = "Aa1531673045!"

BASE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1"


def get_access_token(email, password):
    url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    data = {
        "client_id": "cdse-public",
        "username": email,
        "password": password,
        "grant_type": "password",
    }
    r = requests.post(url, data=data)
    r.raise_for_status()
    return r.json()["access_token"]


def parse_s1_tiff_name(name):
    name = name.lower()
    m = re.match(
        r"(s1[ab])-(iw)-grd-(vv|vh|dv)-(\d{8}t\d{6})-(\d{8}t\d{6})",
        name,
    )
    if not m:
        return None
    platform, _, _, t_start, t_stop = m.groups()
    return platform.upper(), t_start.upper(), t_stop.upper()


def query_product_with_attributes(platform, t_start, t_stop, token):
    headers = {"Authorization": f"Bearer {token}"}

    flt = (
        f"contains(Name,'{platform}') and "
        f"contains(Name,'GRDH') and "
        f"contains(Name,'{t_start}') and "
        f"contains(Name,'{t_stop}')"
    )

    r = requests.get(
        f"{BASE_URL}/Products",
        params={
            "$filter": flt,
            "$top": "1",
            "$expand": "Attributes",
        },
        headers=headers,
        timeout=30,
    )

    r.raise_for_status()

    vals = r.json().get("value", [])
    if not vals:
        return None

    prod = vals[0]

    orbit = None
    for a in prod.get("Attributes", []):
        if a.get("Name") == "orbitDirection":
            orbit = a.get("Value")

    return {
        "product_id": prod.get("Id"),
        "product_name": prod.get("Name"),
        "footprint": prod.get("Footprint"),
        "orbit": orbit,
    }


def write_metadata_to_tif(tif_path, info):
    tags = {
        "ProductID": str(info["product_id"]),
        "ProductName": info["product_name"],
        "OrbitDirection": str(info["orbit"]),
    }

    with rasterio.open(tif_path, "r+") as ds:
        ds.update_tags(**tags)

    print(
        f"✔ 写入完成 | {os.path.basename(tif_path)} | Orbit={info['orbit']}"
    )


def main():
    token = get_access_token(EMAIL, PASSWORD)

    tiffs = [
        f for f in os.listdir(TIFF_FOLDER)
        if f.lower().endswith((".tif", ".tiff"))
    ]

    print(f"发现 {len(tiffs)} 个 TIFF")

    for i, tiff in enumerate(tiffs, 1):
        print(f"[{i}/{len(tiffs)}] 处理 {tiff}")

        info = parse_s1_tiff_name(tiff)
        if not info:
            print("⚠ 文件名解析失败")
            continue

        platform, t_start, t_stop = info

        try:
            prod = query_product_with_attributes(
                platform, t_start, t_stop, token
            )
            if prod is None:
                print("⚠ 未找到匹配产品")
                continue

            write_metadata_to_tif(
                os.path.join(TIFF_FOLDER, tiff), prod
            )

        except Exception as e:
            print("❌ 失败:", e)


if __name__ == "__main__":
    main()
