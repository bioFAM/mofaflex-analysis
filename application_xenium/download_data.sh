#!/bin/bash
set -e  # stop on error
set -o pipefail

# Create base data directory
mkdir -p data

##############################
# Cell type annotations
##############################
echo "Downloading cell type annotations..."
pushd data > /dev/null
wget -q --show-progress https://cdn.10xgenomics.com/raw/upload/v1695234604/Xenium%20Preview%20Data/Cell_Barcode_Type_Matrices.xlsx -O celltypes.xlsx
popd > /dev/null

##############################
# Visium
##############################
echo "Downloading Visium data..."
mkdir -p data/visium
pushd data/visium > /dev/null

wget -q --show-progress https://cf.10xgenomics.com/samples/spatial-exp/2.0.0/CytAssist_FFPE_Human_Breast_Cancer/CytAssist_FFPE_Human_Breast_Cancer_filtered_feature_bc_matrix.h5 -O filtered_feature_bc_matrix.h5

wget -q --show-progress https://cf.10xgenomics.com/samples/spatial-exp/2.0.0/CytAssist_FFPE_Human_Breast_Cancer/CytAssist_FFPE_Human_Breast_Cancer_spatial.tar.gz -O spatial.tar.gz
tar -xzf spatial.tar.gz
rm spatial.tar.gz

popd > /dev/null

##############################
# Xenium
##############################
echo "Downloading Xenium data (this may take several minutes)..."
mkdir -p data/xenium
pushd data/xenium > /dev/null

wget -q --show-progress https://cf.10xgenomics.com/samples/xenium/1.0.1/Xenium_FFPE_Human_Breast_Cancer_Rep1/Xenium_FFPE_Human_Breast_Cancer_Rep1_outs.zip -O Xenium_outs.zip

# Extract within current directory
unzip -q Xenium_outs.zip
rm Xenium_outs.zip

# Move files out of the outs/ folder
mv outs/* .
rm -rf outs

# Decompress cell metadata
gunzip -f cells.csv.gz

# Keep only required files
find . -type f -not -name 'cell_feature_matrix.h5' -not -name 'cells.csv' -delete
rm -rf analysis cell_feature_matrix

popd > /dev/null

##############################
# Chromium
##############################
echo "Downloading Chromium data..."
mkdir -p data/chromium
pushd data/chromium > /dev/null

wget -q --show-progress https://cf.10xgenomics.com/samples/cell-exp/7.0.1/Chromium_FFPE_Human_Breast_Cancer_Chromium_FFPE_Human_Breast_Cancer/Chromium_FFPE_Human_Breast_Cancer_Chromium_FFPE_Human_Breast_Cancer_count_sample_filtered_feature_bc_matrix.h5 -O filtered_feature_bc_matrix.h5

popd > /dev/null

echo "✅ All datasets downloaded successfully!"