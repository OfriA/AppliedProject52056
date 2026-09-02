# AppliedProject52056

## Overview

This repository contains our applied Master's project at **The Hebrew University of Jerusalem**.

The project focuses on the analysis of textual and psychological data related to parental emotion regulation, PTSD, and family well-being during wartime, building on two academic studies:

* **Keleynikov et al. (2025)** — *Trait and State Emotion Regulation and Parental Wellbeing During War*
* **Keleynikov et al. (2025)** — *Parental PTSD and Children’s Well-Being During Wartime: The Role of Interpersonal Emotion Regulation*

Our project examines whether information contained in parents' free-text responses can help predict **child CBCL severity**, represented by three levels: **Low, Medium, and High**.

We combine exploratory data analysis, natural language processing, statistical modeling, and transformer-based methods to study the relationship between parental language and child behavioral and emotional outcomes.

---

## Objectives

* Explore the textual, demographic, and psychological characteristics of the dataset.
* Analyze patterns in parent-written free-text responses.
* Predict child CBCL severity using parental text.
* Compare text-only models with models that also include child age.
* Explore both standard multiclass and ordinal classification approaches.

---

## Dataset Description

The repository includes two primary data files:

| File                      | Description                                                                                                                                           |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`ER_data.xlsx`**        | Contains the main dataset used for analysis, including parental responses, psychological measures, demographic variables, and CBCL-related variables. |
| **`column_details.xlsx`** | Provides descriptions of the variables and columns included in `ER_data.xlsx`.                                                                        |

---

## Methods

The project includes:

1. **Exploratory Data Analysis**

   * Sample and CBCL exploration.
   * Analysis of text characteristics and word frequencies.
   * Hebrew text preprocessing and lemmatization.

2. **Baseline Modeling**

   * Multilingual E5 text embeddings.
   * Multinomial Logistic Regression.

3. **Transformer-Based Modeling**

   * XLM-RoBERTa classification.
   * Evaluation of text-only and text + child-age configurations.
   * Ordinal classification using CORN.

---

## Repository Structure

```text
AppliedProject52056/
│
├── README.md
├── requirements.txt
├── final_report.pdf
│
├── data/
│   ├── ER_data.xlsx
│   └── column_details.xlsx
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_baseline_model.ipynb
│   └── 03_transformer_model_and_corn.ipynb
│
└── src/
    ├── config.py
    ├── data_utils.py
    ├── embeddings.py
    ├── experiments.py
    ├── modeling.py
    ├── transformers_modeling.py
    └── transformers_corn_modeling.py
```

---

## Team

* **Ofri Ahiel**
* **Sagi Levin**

Master's in Data Science, The Hebrew University of Jerusalem
Applied Project - 52056

---

## References

* Keleynikov, M. et al. (2025). *Trait and State Emotion Regulation and Parental Wellbeing During War.* *Personality and Individual Differences.*
* Keleynikov, M. et al. (2025). *Parental PTSD and Children’s Well-Being During Wartime: The Role of Interpersonal Emotion Regulation.* *International Journal on Child Maltreatment.*
