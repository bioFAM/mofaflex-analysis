import gdown
import pandas as pd
import anndata as ad


def load_kang() -> ad.AnnData:
    url = "https://drive.google.com/uc?id=1t3oMuUfueUz_caLm5jmaEYjBxVNSsfxG"
    output = "kang_tutorial.h5ad"
    gdown.download(url, output, quiet=False)

    return ad.read_h5ad("kang_tutorial.h5ad").copy()
