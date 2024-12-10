# toolwear-regressor

This analysis is based on data from the paper:  
**"The multi sensor-based machining signal fusion to compare the relative efficacy of machine learning-based tool wear models"**  
Authored by **Pramod A., Deepak Lawrence K., and Jose Mathew**. Published in **Harvard Dataverse (2022)**.  
[DOI: 10.7910/DVN/7IAJWU](https://doi.org/10.7910/DVN/7IAJWU)

## Overview
- **Objective**: To define the anomaly range in tool wear using 12 variables derived from 7 types of sensory data.
- **Labels**:
  - **Continuous tool wear values**: Derived using label augmentation techniques by fitting discrete labels through a defined function.


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
