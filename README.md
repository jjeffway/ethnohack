# ethnohack
Repository for EthnoHack 2026 Data Hackathon on Ethnopharmacology &amp; Health submission

## gameplan
1. Choose two datasets for complementary targets (active/inactive compounds)
2. Train tree-ensemble model on each dataset (including hyperparameter optimization), use scaffold split and perform balancing
3. Evaluate each model separately (ROC-AUC, PR-AUC, balanced accuracy)
4. Predict natural products against both targets
5. Run TreeSHAP and LIME explainers to identify locally and globally important features
6. Use both probabilities to find compounds from the same plant that hit complementary targets