"""MovieLens 数据下载"""
import urllib.request
import zipfile
import os


def download_ml1m(save_dir="data/raw/ml-1m"):
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    zip_path = os.path.join(save_dir, "ml-1m.zip")
    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(os.path.join(save_dir, "ml-1m", "ratings.dat")):
        print("MovieLens 1M 已存在，跳过下载")
        return

    print("正在下载 MovieLens 1M ...")
    urllib.request.urlretrieve(url, zip_path)
    print("正在解压 ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(save_dir)
    os.remove(zip_path)
    print(f"完成！数据位于 {save_dir}/ml-1m/")


if __name__ == "__main__":
    download_ml1m()
