# 🛠️ Toolwear Regressor

## 🔍 Overview

**Objective:**  
To advance the precision of tool wear prediction by augmenting temporal granularity, refining label structures, and integrating domain-specific insights.

This repository provides an end-to-end pipeline for tool wear value prediction, incorporating innovative data augmentation strategies and refined labeling schemes. The project lays the foundation for scalable, real-time diagnostics and will be extended into a multimodal system combining both temporal and acoustic signals for robust stage-wise tool condition monitoring.

---

## ⚙️ Key Contributions

- **High-Resolution Temporal Augmentation**  
  Transformed short-duration discrete data into high-frequency sequences sampled at **0.1-second intervals**, enabling real-time inference and finer-grained predictive modeling.

- **Label Enrichment**  
  Original discrete wear-level labels were transformed into **continuous regression targets** through domain-informed interpolation, enhancing model learning dynamics.

- **Feature and Label Expansion**  
  Leveraged EDA and domain knowledge to introduce **categorical phase labels** and **extended tool wear trajectories**, improving both the interpretability and robustness of downstream models.

- **Accurate Regression Modeling**  
  Implemented and evaluated several regression models using the augmented dataset. These models were fine-tuned to leverage both **numerical** and **categorical context** for improved performance and generalizability.

---

## 🧪 Data Preprocessing

- **Source**: Raw data from the referenced study was used as the foundation.

- **Preprocessing Steps**:
  - **Temporal Resolution Enhancement**: Augmented tool wear readings at a **0.1s interval** for real-time application.
  - **Label Continuity**: Transformed discrete stage labels into **smooth, continuous values** for regression modeling.
  - **Phase Classification**: Assigned additional categorical phase labels to reflect wear progression stages (e.g., Initial, Steady, Failure).

---

## 🤖 Regression Modeling

- Constructed predictive models for `tool_wear_value` based on the enriched dataset.
- Evaluated models using standard metrics (e.g., MAE, RMSE) and cross-validation to ensure consistency across wear phases.
- Incorporated **hybrid feature sets** (e.g., statistical, temporal, label-derived) to enhance model expressiveness.

---

## 🔁 Planned Extension: Multimodal Stage Diagnosis

As the next phase of this project, we plan to expand into a **multimodal diagnostic system** by integrating **acoustic emission (AE) signals** with existing time-series data:

- AE data will be processed into **Mel-Spectrogram representations**, allowing deep models (e.g., CNNs or vision transformers) to capture frequency-domain patterns.
- The combined temporal–acoustic pipeline will enable **stage-wise diagnosis** of tool wear states and transitions, improving both prediction accuracy and explainability.
- This fusion approach aims to bridge sensor modalities, aligning temporal degradation patterns with sound-based signal signatures for a holistic condition monitoring service.





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
