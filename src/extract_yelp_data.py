import os, tarfile
from pathlib import Path

def extract_yelp_data(tar_path,extract_path):
    """
    Extract Yelp dataset .tar file
    :param tar_path:
    :param extract_path:
    """
    tar = Path(os.path.expanduser(tar_path))
    extract_path = Path(os.path.expanduser(extract_path))
    extract_path.mkdir(parents=True, exist_ok=True)

    #Extract
    with tarfile.open(tar) as tar:
        tar.extractall(path = extract_path)
        print(f"Extracted {tar.name} to {extract_path}")

if __name__ == "__main__":
    Tar_Path = "~/Documents/cs_ml/capstone/data/raw/yelp_dataset.tar"
    Extract_Path = "~/Documents/cs_ml/capstone/data/extracted"
    extracted_files = extract_yelp_data(Tar_Path,Extract_Path)