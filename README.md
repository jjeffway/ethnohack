# EthnoHack: Structure-Based QSAR Models for COX-2 and 5-LOX

**Ethno_PAIR** is a computational chemistry project that builds structure-based Quantitative Structure-Activity Relationship (QSAR) models to identify natural products with synergistic anti-inflammatory activity. The project targets two key enzymes in the inflammatory cascade: **cyclooxygenase-2 (COX-2)** and **5-lipoxygenase (5-LOX, ALOX5)**. These enzymes represent complementary pathways in the arachidonic acid cascade, and their simultaneous inhibition may provide enhanced anti-inflammatory effects with reduced side effects compared to single-target drugs (Chatzipieris et al., 2026).

## Biological Context

### Inflammatory Pathways

Inflammation is a complex biological response involving multiple pathways. Two critical enzymes in the arachidonic acid metabolism pathway are:

- **COX-2 (Cyclooxygenase-2, CHEMBL230)**: Converts arachidonic acid to prostaglandins, which mediate pain, fever, and inflammation. COX-2 is inducible and expressed at sites of inflammation (Smith et al., 2000). Meta-analyses of randomised trials show that all NSAIDs raise the risk of upper gastrointestinal complications, while selective COX-2 inhibitors (coxibs) and high-dose diclofenac also increase major vascular events (Coxib and traditional NSAID Trialists' [CNT] Collaboration, 2013).

- **5-LOX (5-Lipoxygenase, ALOX5, CHEMBL215)**: Converts arachidonic acid to leukotrienes, which are potent inflammatory mediators involved in asthma, allergic reactions, and chronic inflammation (Rådmark et al., 2015; Haeggström, 2018). Leukotrienes have an established pathogenic role in allergic airway disease, and the 5-LOX inhibitor zileuton is approved in the US for asthma; leukotrienes have also been linked to low-grade inflammation in cardiovascular, metabolic, and neurodegenerative disease (Haeggström, 2018).

### Synergistic Inhibition Hypothesis

The **synergistic inhibition hypothesis** posits that compounds targeting both COX-2 and 5-LOX simultaneously may provide:
- **Enhanced efficacy**: Blocking both prostaglandin and leukotriene production (Chatzipieris et al., 2026)
- **Reduced side effects**: Lower doses of each inhibitor may be effective
- **Broader spectrum**: Addressing multiple inflammatory mediators
- **Natural product inspiration**: Many medicinal plants contain compounds that modulate both the COX and the LOX branch of the cascade; metabolomic screening of 57 Asteraceae leaf extracts identified constituents with dual COX-1/5-LOX inhibitory potential (Chagas-Paula et al., 2015)

Note that these are the design rationales motivating dual-inhibitor development, not established clinical outcomes. Dual COX/5-LOX inhibition remains an active area of medicinal chemistry rather than a validated therapeutic class (Chatzipieris et al., 2026).

This approach is particularly relevant for **ethnopharmacology** — the study of traditional medicine systems, where plant extracts often contain multiple bioactive compounds that work synergistically. Combination effects in such extracts can be synergistic or antagonistic, and both are documented, so a multi-compound extract is not automatically more active than its purified constituents (Caesar & Cech, 2019).

## Pipeline Overview

The pipeline consists of four main stages:

1. **Data Download**: Download raw ChEMBL bioactivity data using `download_chembl.py`
2. **Data Processing**: Clean, binarize, compute features, and prune using `data_pipeline.py`
3. **Model Training**: Train Random Forest classifiers with scaffold-based splitting using `cox_train.py` and `lox5_train.py`
4. **Scoring**: Score COCONUT natural products using `score_both.py`


## Computational Approach

### Structure-Based QSAR Modeling

This project employs **structure-based QSAR** modeling, which predicts biological activity from molecular structure alone, using calculated rather than experimentally measured physicochemical properties. The models use:

- **Molecular descriptors**: 2D physicochemical properties computed from SMILES strings using RDKit
- **Binary classification**: Active (1) vs inactive (0) based on pChEMBL threshold
- **Tree ensemble methods**: Random Forest classifiers with hyperparameter optimization
- **Scaffold-based validation**: Evaluation using Bemis-Murcko scaffold splits to reduce analogue bias

### Limitations of the validation strategy

Scaffold splits reduce analogue bias but do not remove it. Scaffolds with different names can be structurally near-identical, so highly similar molecules may still be distributed across training and test sets, and reported performance can remain optimistic relative to a prospective screen (Guo et al., 2024, preprint). Model performance should therefore be read alongside a nearest-neighbour similarity baseline and an applicability-domain check, rather than as a single accuracy figure.

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

Then follow [these Instructions](./datasets_preparation/README.md) for downloading and preparing the data.

### Authors and acknowledgment
Chiara Marie Dassow - model training <br>
Jenna Jeffway - explainable AI <br>
Daria Kochenko - data fetching <br>

### AI Usage in this Project
Claude Opus 5.5 and Mistral Medium 3.5 were used for code generation, bug fixing, restructuring and text generation. Sources were manually controlled.


## License
Bioactivity data taken from ChEMBL database which is licensed by CC BY-SA 3.0.

## References

Caesar, L. K., & Cech, N. B. (2019). Synergy and antagonism in natural product extracts: When 1 + 1 does not equal 2. *Natural Product Reports, 36*(6), 869–888. https://doi.org/10.1039/C9NP00011A

Chagas-Paula, D. A., Zhang, T., Da Costa, F. B., & Edrada-Ebel, R. (2015). A metabolomic approach to target compounds from the Asteraceae family for dual COX and LOX inhibition. *Metabolites, 5*(3), 404–430. https://doi.org/10.3390/metabo5030404

Chatzipieris, F. P., Petsas, E., Lambrinidis, G., Vassiliou, S., & Chasapis, C. T. (2026). Recent advances in dual COX/LOX inhibitor design (2020–2024): Establishing "the rule of four for inflammation". *Life, 16*(1), 163. https://doi.org/10.3390/life16010163

Coxib and traditional NSAID Trialists' (CNT) Collaboration. (2013). Vascular and upper gastrointestinal effects of non-steroidal anti-inflammatory drugs: Meta-analyses of individual participant data from randomised trials. *The Lancet, 382*(9894), 769–779. https://doi.org/10.1016/S0140-6736(13)60900-9

Guo, Q., Hernández-Hernández, S., & Ballester, P. J. (2024). *Scaffold splits overestimate virtual screening performance* [Preprint]. arXiv. https://arxiv.org/abs/2406.00873

Haeggström, J. Z. (2018). Leukotriene biosynthetic enzymes as therapeutic targets. *The Journal of Clinical Investigation, 128*(7), 2680–2690. https://doi.org/10.1172/JCI97945

Rådmark, O., Werz, O., Steinhilber, D., & Samuelsson, B. (2015). 5-Lipoxygenase, a key enzyme for leukotriene biosynthesis in health and disease. *Biochimica et Biophysica Acta (BBA) – Molecular and Cell Biology of Lipids, 1851*(4), 331–339. https://doi.org/10.1016/j.bbalip.2014.08.012

Smith, W. L., DeWitt, D. L., & Garavito, R. M. (2000). Cyclooxygenases: Structural, cellular, and molecular biology. *Annual Review of Biochemistry, 69*, 145–182. https://doi.org/10.1146/annurev.biochem.69.1.145
