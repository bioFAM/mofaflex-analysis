#!/bin/bash

# cell type annotations
mkdir data
pushd data
wget https://cdn.10xgenomics.com/raw/upload/v1695234604/Xenium%20Preview%20Data/Cell_Barcode_Type_Matrices.xlsx
mv Cell_Barcode_Type_Matrices.xlsx celltypes.xlsx
popd

# Visium
mkdir -p data/visium
pushd data/visium
wget https://cf.10xgenomics.com/samples/spatial-exp/2.0.0/CytAssist_FFPE_Human_Breast_Cancer/CytAssist_FFPE_Human_Breast_Cancer_filtered_feature_bc_matrix.h5
wget https://cf.10xgenomics.com/samples/spatial-exp/2.0.0/CytAssist_FFPE_Human_Breast_Cancer/CytAssist_FFPE_Human_Breast_Cancer_spatial.tar.gz
tar -xzf CytAssist_FFPE_Human_Breast_Cancer_spatial.tar.gz
rm CytAssist_FFPE_Human_Breast_Cancer_spatial.tar.gz
mv CytAssist_FFPE_Human_Breast_Cancer_filtered_feature_bc_matrix.h5 filtered_feature_bc_matrix.h5
popd

# Xenium
mkdir -p data/xenium
pushd data/xenium
wget https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_outs.zip
unzip -j Xenium_FFPE_Human_Breast_Cancer_Rep1_outs.zip outs/cell_feature_matrix.h5 outs/cells.csv.gz
gunzip cells.csv.gz
rm Xenium_FFPE_Human_Breast_Cancer_Rep1_outs.zip
popd

# Chromium
mkdir -p data/chromium
pushd data/chromium
wget https://cf.10xgenomics.com/samples/cell-exp/7.0.1/Chromium_FFPE_Human_Breast_Cancer_Chromium_FFPE_Human_Breast_Cancer/Chromium_FFPE_Human_Breast_Cancer_Chromium_FFPE_Human_Breast_Cancer_count_sample_filtered_feature_bc_matrix.h5
mv Chromium_FFPE_Human_Breast_Cancer_Chromium_FFPE_Human_Breast_Cancer_count_sample_filtered_feature_bc_matrix.h5 filtered_feature_bc_matrix.h5
popd

    
