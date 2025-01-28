# run with "muon" kernel

def main():
    import muon as mu
    import pandas as pd
    from data_loader import load_cll

    data = load_cll()
    mdata = mu.MuData(data)
    obs = pd.read_csv("data/cll_metadata.csv", index_col="Sample")
    mdata.obs = mdata.obs.join(obs)

    mu.tl.mofa(mdata, use_obs='union',
        n_factors=10, convergence_mode='medium',
        outfile="mofa.h5"
        )
    
if __name__ == "__main__":
    main()
    
