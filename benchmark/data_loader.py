import gdown
from pathlib import Path


def load_kang() -> None:
    """Load the Kang et al. dataset from Google Drive."""
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    output_path = data_dir / "kang_tutorial.h5ad"
    url = "https://drive.google.com/uc?id=1t3oMuUfueUz_caLm5jmaEYjBxVNSsfxG"
    gdown.download(url, str(output_path), quiet=False)


if __name__ == "__main__":
    adata = load_kang()