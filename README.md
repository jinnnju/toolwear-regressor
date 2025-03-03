# toolwear-regressor


## Overview
- **Objective**: To enhance tool wear prediction by augmenting the original dataset and refining the labeling process.
- **Key Enhancements**:
  - Augmented short-duration data into real-time 0.1s interval data.
  - Expanded the existing tool-wear values for more precise predictions.
  - Developed a more accurate tool wear prediction model by leveraging domain knowledge and exploratory data analysis (EDA).
  - Introduced additional categorical labels to improve experimental robustness and analysis.

This repository contains code implementing these enhancements to create a more refined tool wear prediction framework.


## Data Preprocessing
- Raw data proposed in the referenced study was utilized.
- Preprocessing steps:
  1. **Label Augmentation**: Fitting discrete labels into a continuous label representation.
  2. **Temporal Resolution**: Processed the data at a 0.1-second interval to enable real-time inference.

## Regression Models
- Developed regression models to predict `tool_wear_value` using the preprocessed data.
- Models leverage the augmented continuous labels and categorical classifications for enhanced performance.

## Citation
```bibtex
@data{DVN/7IAJWU_2022,
  author = {Pramod A and Deepak Lawrence K and Jose Mathew},
  publisher = {Harvard Dataverse},
  title = {{The multi sensor-based machining signal fusion to compare the relative efficacy of machine learning based tool wear models}},
  UNF = {UNF:6:dXrPQBleR2MIYhnAnGNFJQ==},
  year = {2022},
  version = {V1},
  doi = {10.7910/DVN/7IAJWU},
  url = {https://doi.org/10.7910/DVN/7IAJWU}
}
