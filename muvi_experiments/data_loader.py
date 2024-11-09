import gdown
import pandas as pd
import anndata as ad
import os
from pathlib import Path

def load_kang():
    # Create data directory if it doesn't exist
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    # Set output path inside data directory
    output_path = data_dir / "kang_tutorial.h5ad"
    
    # Download URL
    url = "https://drive.google.com/uc?id=1t3oMuUfueUz_caLm5jmaEYjBxVNSsfxG"
    
    # Download the file
    gdown.download(url, str(output_path), quiet=False)


if __name__ == "__main__":
    adata = load_kang()
