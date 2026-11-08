# Data preparation

## Data sources

For models training activity datasets for targets [CHEMBL215](https://www.ebi.ac.uk/chembl/explore/target/CHEMBL215) and [CHEMBL230](https://www.ebi.ac.uk/chembl/explore/target/CHEMBL230) were used.

## Dependencies

The data preparation scripts require:

- Python >= 3.13
- pandas >= 3.1.0
- numpy >= 2.5.4
- rdkit >= 2026.9.1
- chembl-webresource-client >= 0.10.9 (for download_chembl.py only)

## Fast start

To download the raw csv files of the databases use
```bash
python datasets_preparation/download_chembl.py CHEMBL230 -o datasets_preparation/cox2.csv

python datasets_preparation/download_chembl.py CHEMBL215 -o datasets_preparation/5lox.csv
```
To prepare the dataset for the model the NaN-containing rows were filtered out, features chosen depending on correlation, molecular descriptors calculated using RDKit. Dataset processing is performed with
``` bash
python datasets_preparation/data_pipeline.py --output datasets_preparation/cox2_model_table.csv datasets_preparation/cox2.csv

python datasets_preparation/data_pipeline.py --output datasets_preparation/5lox_model_table.csv datasets_preparation/5lox.csv
```
## Pipeline Overview

The data preparation pipeline performs the following steps:

1. **Download**: Retrieve raw bioactivity data from ChEMBL
2. **Clean**: Filter columns, remove null pChEMBL values, group by molecule, aggregate with median
3. **Binarize**: Convert pChEMBL values to binary labels (active/inactive) using threshold 6.0
4. **Compute Features**: Calculate ~200+ RDKit molecular descriptors
5. **Prune Features**: Remove correlated features ( Spearman ρ| > 0.5)
6. **Build Model Table**: Create final table with ChEMBL_IDs, scaffold, Activity, and selected features
7. **Merge Stereoisomers**: Combine stereoisomers into one row per flat structure


### Feature Selection

- **Core features** (always included): MolWt, FractionCSP3, TPSA, BalabanJ, BertzCT, Kappa1-3, MolLogP
- **Additional features**: Chi indices (0-4, normal and non-normal), EState_VSA (1-11), fragment counts (fr_* descriptors)
- **Pruning logic**: Features with  Spearman |ρ| > 0.5 are considered redundant. When two features are correlated:
  - If one is protected, keep the protected one
  - Otherwise, keep the one with higher activity correlation (|r| with Activity)
  - If tie (< 0.02 difference), use redundancy score as tie-breaker

## Some notes

Thresholds for binarization and pruning were chosen arbitrarily. <br>
For tighter or looser thresholding tune the value of <br> `BINARY_THRESHOLD` (line 53 in data_pipeline.py) and<br> `PRUNING_THRESHOLD` (line 78 in data_pipeline.py)