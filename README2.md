# CBCL Severity Prediction from Parent Free-Text

This repository contains the code and analysis for an applied data science project on predicting child CBCL severity from parent-written free text.

The prediction target is child CBCL severity, represented by three classes:

* **Low**
* **Medium**
* **High**

The project investigates whether parent-written descriptions of stressful events and emotion-regulation experiences contain information that can help identify children with elevated behavioral and emotional difficulties.

Two main feature configurations are considered:

1. Parent free-text only (`Event` + `EER_text`)
2. Parent free-text + child age

The primary evaluation metrics are **Macro F1** and **High-class F2**, with additional emphasis on recall for the High-CBCL group.

## Project Workflow

The analysis is organized into three main notebooks.

### 1. Exploratory Data Analysis

`notebooks/01_exploratory_data_analysis.ipynb`

Explores the study sample and modeling variables, including:

* sample construction and CBCL score calculation,
* CBCL score distributions,
* selected demographic relationships,
* characteristics of the parent-written text,
* Hebrew text preprocessing and lemmatization,
* overall lexical patterns,
* word-frequency differences across CBCL severity groups.

### 2. Baseline Model

`notebooks/02_baseline_model.ipynb`

Implements the baseline prediction pipeline using:

* multilingual E5 sentence embeddings (`intfloat/multilingual-e5-base`),
* separate embeddings of `Event` and `EER_text`,
* concatenated text representations,
* multinomial Logistic Regression.

Both text-only and text + child-age feature configurations are evaluated.

The notebook also includes permutation-based null-distribution analysis and sensitivity analysis.

### 3. Transformer Model and Research Extension

`notebooks/03_transformer_model_and_corn.ipynb`

Develops the transformer-based prediction pipeline using XLM-RoBERTa.

The model-development process evaluates:

* initial full fine-tuning,
* a frozen transformer backbone,
* class rebalancing through oversampling,
* moderate oversampling,
* incorporation of child age as structured metadata.

The final multiclass model uses a frozen XLM-RoBERTa backbone with moderate oversampling and child age.

The notebook also includes:

* quantitative null-distribution analysis,
* qualitative error analysis focused on the High-CBCL class,
* an ordinal-classification research extension using CORN (Conditional Ordinal Regression for Neural Networks).

## Repository Structure

```text
AppliedProject52056/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── ER_data.xlsx
│   └── Column_details.xlsx
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

## Source Modules

* `config.py` – central project configuration and modeling constants.
* `data_utils.py` – dataset preparation, CBCL score construction, train/test splitting, and severity-label creation.
* `embeddings.py` – multilingual E5 embedding generation.
* `modeling.py` – baseline Logistic Regression models and evaluation utilities.
* `experiments.py` – reusable experimental and permutation-analysis utilities.
* `transformers_modeling.py` – XLM-RoBERTa multiclass modeling pipeline.
* `transformers_corn_modeling.py` – ordinal CORN transformer implementation.

## Data

`ER_data.xlsx` contains the study dataset used throughout the project.

`Column_details.xlsx` contains information describing the dataset variables and columns.

## Installation

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

The transformer and sentence-embedding models are downloaded automatically from Hugging Face when first used.

The exploratory text analysis also downloads the Hebrew Stanza language resources when required.

## Running the Project

The notebooks are intended to be followed in numerical order:

```text
01_exploratory_data_analysis.ipynb
02_baseline_model.ipynb
03_transformer_model_and_corn.ipynb
```

Reusable preprocessing and modeling functionality is maintained in the `src/` directory rather than duplicated across notebooks.
