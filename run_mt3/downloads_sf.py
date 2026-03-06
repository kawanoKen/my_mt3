import urllib.request
import zipfile
import os

def download_soundfont():
    # 軽量で高品質な「GeneralUser GS」のURL（zip形式）
    url = "https://schristiancollins.com/soundfonts/GeneralUser_GS_1.471.zip"
    zip_path = "GeneralUser_GS.zip"
    extract_path = "./"

    print("SoundFontをダウンロード中... (約30MB)")
    urllib.request.urlretrieve(url, zip_path)

    print("解凍中...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # zip内にある .sf2 ファイルを探して解凍
        for file in zip_ref.namelist():
            if file.endswith(".sf2"):
                zip_ref.extract(file, extract_path)
                # 使いやすいようにファイル名をリネーム
                os.rename(os.path.join(extract_path, file), "GeneralUser_GS.sf2")
                print(f"完了！ 保存先: {os.path.abspath('GeneralUser_GS.sf2')}")
                break
    
    # 不要になったzipファイルを削除
    os.remove(zip_path)

if __name__ == "__main__":
    download_soundfont()