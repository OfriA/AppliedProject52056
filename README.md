# AppliedProject52056

## Overview
This repository contains our applied Master's project at The Hebrew University of Jerusalem.  
The project focuses on **text analysis related to parental emotion regulation and PTSD**, inspired by two academic studies:

- **Keleynikov et al. (2025)** — *Trait and State Emotion Regulation and Parental Wellbeing During War*
- **Keleynikov et al. (2025)** — *Parental PTSD and Children’s Well-Being During Wartime: The Role of Interpersonal Emotion Regulation*

Our goal is to explore new insights from textual and psychological data that were previously analyzed in these studies — particularly how parents’ language, emotional expression, and coping strategies relate to mental health and family outcomes during wartime.  

We plan to use **natural language processing (NLP)** and **statistical analysis** to extend the original findings and derive new conclusions from the text data.

---

## Objectives
- Analyze text responses from parents regarding stress, emotion regulation, and well-being.
- Identify linguistic and psychological markers of adaptive vs. maladaptive emotion regulation.
- Explore relationships between linguistic features, PTSD symptoms, and parental burnout.
- Generate new hypotheses about emotion regulation and resilience during wartime.

---

## Dataset Description
The repository includes two primary data files:

| File | Description |
|------|--------------|
| **`ER_data.xlsx`** | Contains the main dataset used for analysis — parental responses, emotional measures, and related variables. |
| **`column_details.xlsx`** | Provides a description of each column in `ER_data.xlsx`, including variable names, types, and measurement details. |

These datasets are based on or inspired by data from the two referenced studies and will serve as the foundation for text and statistical analysis.

---

## Methods (Planned)
1. **Data Preprocessing**
   - Cleaning, tokenization, and normalization of text fields.
   - Handling missing or outlier data.

2. **Textual Analysis**
   - Sentiment and emotion detection using lexicons or pretrained models.
   - Topic modeling (e.g., LDA, BERTopic) to uncover themes.
   - Linguistic feature extraction (e.g., LIWC categories, pronoun use, word embeddings).

3. **Quantitative Analysis**
   - Correlation and regression between linguistic markers and psychological variables (e.g., PTSD, parental burnout).
   - Group comparisons (e.g., high vs. low emotion regulation).
   - Visualization of key emotional and cognitive patterns.

4. **Advanced Modeling (optional)**
   - Fine-tuned transformer-based models (e.g., BERT, DistilBERT).
   - Clustering or classification of coping styles.

---

## Expected Outcomes
- Identification of emotional and linguistic indicators of stress and coping.
- Quantitative insights linking language use to well-being measures.
- Visualization of emotion regulation dynamics in wartime family contexts.
- New hypotheses and intervention ideas for supporting parents and children under trauma exposure.

---

##Repository Structure
📂 data/
┣ 📜 ER_data.xlsx
┗ 📜 column_details.xlsx
📂 notebooks/
┗ 📓 text_analysis.ipynb # Jupyter notebooks for exploration & modeling
📂 src/
┣ 📜 preprocessing.py
┣ 📜 analysis.py
┗ 📜 visualization.py
📂 reports/
┗ 📊 figures, charts, and summary outputs
README.md


---

## Team
- **Ofri Ahiel**
- **Sagi Levin**  
Master’s in Data Science, The Hebrew University of Jerusalem  
Applied Project - 52056

---

## References
- Keleynikov, M. et al. (2025). *Trait and State Emotion Regulation and Parental Wellbeing During War.* *Personality and Individual Differences.*
- Keleynikov, M. et al. (2025). *Parental PTSD and Children’s Well-Being During Wartime: The Role of Interpersonal Emotion Regulation.* *International Journal on Child Maltreatment.*
