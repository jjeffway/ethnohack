# EthnoHack: Structure-Based QSAR Models for COX-2 and 5-LOX

**JOIN_CL** is a computational chemistry project that builds structure-based Quantitative Structure-Activity Relationship (QSAR) models to identify natural products with synergistic anti-inflammatory activity. The project targets two key enzymes in the inflammatory cascade: **cyclooxygenase-2 (COX-2)** and **5-lipoxygenase (5-LOX, ALOX5)**. These enzymes represent complementary pathways in the arachidonic acid cascade, and their simultaneous inhibition may provide enhanced anti-inflammatory effects with reduced side effects compared to single-target drugs.

## Biological Context

### Inflammatory Pathways

Inflammation is a complex biological response involving multiple pathways. Two critical enzymes in the arachidonic acid metabolism pathway are:

- **COX-2 (Cyclooxygenase-2, CHEMBL230)**: Converts arachidonic acid to prostaglandins, which mediate pain, fever, and inflammation. COX-2 is inducible and expressed at sites of inflammation. Non-selective COX inhibitors (like traditional NSAIDs) cause gastrointestinal side effects, while selective COX-2 inhibitors may increase cardiovascular risk.

- **5-LOX (5-Lipoxygenase, ALOX5, CHEMBL215)**: Converts arachidonic acid to leukotrienes, which are potent inflammatory mediators involved in asthma, allergic reactions, and chronic inflammation. 5-LOX inhibition can reduce leukocyte recruitment and vascular permeability.

### Synergistic Inhibition Hypothesis

The **synergistic inhibition hypothesis** posits that compounds targeting both COX-2 and 5-LOX simultaneously may provide:
- **Enhanced efficacy**: Blocking both prostaglandin and leukotriene production
- **Reduced side effects**: Lower doses of each inhibitor may be effective
- **Broader spectrum**: Addressing multiple inflammatory mediators
- **Natural product inspiration**: Many traditional medicinal plants contain compounds that modulate both pathways

This approach is particularly relevant for **ethnopharmacology** — the study of traditional medicine systems, where plant extracts often contain multiple bioactive compounds that work synergistically.

## Pipeline Overview

The pipeline consists of four main stages:

1. **Data Download**: Download raw ChEMBL bioactivity data using `download_chembl.py`
2. **Data Processing**: Clean, binarize, compute features, and prune using `data_pipeline.py`
3. **Model Training**: Train Random Forest classifiers with scaffold-based splitting using `cox_train.py` and `lox5_train.py`
4. **Scoring**: Score COCONUT natural products using `score_both.py`


## Computational Approach

### Structure-Based QSAR Modeling

This project employs **structure-based QSAR** modeling, which predicts biological activity based solely on molecular structure without requiring experimental physicochemical properties (logP, pKa, solubility). The models use:

- **Molecular descriptors**: 2D physicochemical properties computed from SMILES strings using RDKit
- **Binary classification**: Active (1) vs inactive (0) based on pChEMBL threshold
- **Tree ensemble methods**: Random Forest classifiers with hyperparameter optimization
- **Scaffold-based validation**: Rigorous evaluation using Murcko scaffold-based splits to prevent analogue bias

## Setup

### Prerequisites

- Python >= 3.13
- Dependencies listed in `pyproject.toml`

### Installation

```bash
# Clone the repository
git clone https://github.com/jjeffway/ethnohack.git
cd ethnohack

# Create virtual environment (using uv)
uv sync

# Or using pip
pip install -e .
```

### Authors and acknowledgment
Chiara Marie Dassow - model training <br>
Jenna Jeffway - explainable AI <br>
Daria Kochenko - data fetching <br>



## License
For open source projects, say how it is licensed.