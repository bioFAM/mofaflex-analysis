import muon as mu
import pandas as pd
from data_loader import load_cll

def main():
    mdata = load_cll()

    mu.tl.mofa(mdata, use_obs='union',
        n_factors=10, convergence_mode='medium',
        outfile="models/mofa.h5"
        )
    
if __name__ == "__main__":
    main()
    
