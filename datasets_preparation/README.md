# Data preparation

## Data sources
For models training activity datasets for targets [CHEMBL215](https://www.ebi.ac.uk/chembl/explore/target/CHEMBL215) and [CHEMBL230](https://www.ebi.ac.uk/chembl/explore/target/CHEMBL230) were used.

## Fast start
To download the raw csv files of the databases use
```bash
python datasets_preparation\\download_chembl.py CHEMBL230 -o datasets_preparation\\cox2.csv

python datasets_preparation\\download_chembl.py CHEMBL215 -o datasets_preparation\\5lox.csv
```
To prepare the dataset for the model the NaN-containing rows were filtered out, features chosen depending on correlation, molecular descriptors calculated using RDKit. Dataset processing is performed with
``` bash
python datasets_preparation\\data_pipeline.py --output datasets_preparation\\cox2_model_table.csv datasets_preparation\\cox2.csv

python datasets_preparation\\data_pipeline.py --output datasets_preparation\\5lox_model_table.csv datasets_preparation\\5lox.csv
```
## Some notes

Thresholds for binarization and pruning were chosen arbitrarily. <br>
For tighter or looser thresholding tune the value of <br> `BINARY_THRESHOLD` (line 53 in data_pipeline.py) and<br> `PRUNING_THRESHOLD` (line 78 in data_pipeline.py)