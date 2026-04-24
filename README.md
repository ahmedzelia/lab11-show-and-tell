# Show & Tell: NLP Classification of Data Stories
### IS 733 — Lab 11 (Part II) | University of Maryland Baltimore County

---

## Overview

This project replicates the NLP classification pipeline from:

> **Sivakumar et al. (2024)** — *Show and Tell: Exploring Large Language Model's Potential in Formative Educational Assessment of Data Stories*
> Presented at the Gen4DS Workshop

The paper explores using **GPT-4o** to automatically assess whether data narratives are at a surface "**Show**" level (describing a chart) or a deeper "**Tell**" level (interpreting and inferring from a chart). This repository contains the full classification pipeline built on the real one-shot GPT-4o dataset, including feature extraction, model training, Leave-One-Plot-Out validation, and Figure 6 replication.

---

## Repository Structure

```
show-and-tell-nlp/
│
├── README.md
├── lab11_notebook.ipynb          # Full pipeline — code + outputs
├── Lab_11_results.pdf            # PDF of executed notebook
└── data_stories_one_shot.csv     # Dataset — 130 GPT-4o sentences (one-shot)
```

---

## Dataset

| Property | Value |
|---|---|
| File | `data_stories_one_shot.csv` |
| Source | GPT-4o, one-shot prompting |
| Sentences | 130 |
| Plots | 12 data visualizations |
| Columns | `Plot_Name`, `Stage`, `Quality`, `Sentence` |

### Label Encoding
| Label | Meaning | Stage | Count |
|---|---|---|---|
| `0` | **Show** | Stage 1 — reading the data (description) | 79 |
| `1` | **Tell** | Stage 2/3 — reading between/beyond the data (interpretation) | 51 |

### Plots Included

| Plot | Difficulty | Chart Type |
|---|---|---|
| Vaccine | Easy | Bar |
| Youtube | Easy | Heatmap |
| Degree | Easy | Bar |
| Meaningful Life | Easy | Dot |
| Solar Activity | Easy/Medium | Line |
| Walk Dog | Medium | Line |
| Hurricane | Medium | Bar |
| Time Use | Medium | Area |
| STEM | Medium/Hard | Bar |
| Wealth Gap | Medium/Hard | Line |
| MAP | Medium/Hard | Dot |
| Vulnerability | Difficult | Dot |

> ⚠️ `time use` was excluded from Leave-One-Plot-Out — it contains 0 Tell sentences, making AUC computation undefined for a single-class fold.

---

## Pipeline

### Text Pre-Processing
Following the exact 5-step pipeline from the paper:
```
1. Lowercase
2. Remove punctuation
3. Tokenize
4. Remove stop words
5. Lemmatize (rule-based suffix stripping)
```

### Feature Extraction
Two feature types combined into a single matrix:

```
TF-IDF (unigrams + bigrams, 500 features, sublinear_tf=True)
+
Hand-crafted NLP features (8 dimensions):
  [0] Tell-marker density
  [1] Show-marker density
  [2] Hedge word count
  [3] Sentence length
  [4] Has comparison language (binary)
  [5] Has causal language (binary)
  [6] Raw Tell word count
  [7] Raw Show word count
```

### Classifiers
| Classifier | Notes |
|---|---|
| Logistic Regression | From paper |
| Naive Bayes | From paper |
| SVM (linear kernel) | From paper |
| Random Forest | Bonus — not in paper |

### Validation Strategies
| Strategy | Description |
|---|---|
| **5-Fold CV** | Random stratified split — may overestimate due to data leakage across plots |
| **Leave-One-Plot-Out** | Train on 11 plots, test on 1 — simulates real classroom deployment |

---

## Results

### 5-Fold Cross-Validation

| Classifier | AUC | ±Std | Accuracy |
|---|---|---|---|
| Logistic Regression | 0.923 | 0.035 | 0.831 |
| Naive Bayes | 0.927 | 0.023 | 0.808 |
| SVM | 0.938 | 0.010 | 0.862 |
| Random Forest *(bonus)* | 0.943 | 0.057 | 0.869 |

> Paper benchmark (One-Shot CV): LR = 0.79 | NB = 0.78 | SVM = 0.79

### Leave-One-Plot-Out (LOPO) — Key Result

| Classifier | AUC | ±Std | Accuracy | Paper LOPO |
|---|---|---|---|---|
| Logistic Regression | 0.942 | 0.066 | 0.802 | 0.92 |
| Naive Bayes | 0.952 | 0.065 | 0.781 | 0.93 |
| **SVM** | **0.956** | 0.070 | 0.803 | **0.94** |
| Random Forest *(bonus)* | 0.922 | 0.074 | 0.775 | N/A |

>  LOPO results match the paper within **±0.02** — strong replication.

### Key Findings

1. **LOPO ≥ CV** — the model generalizes well to charts it has never seen, which is the real-world classroom requirement
2. **No dominant classifier** -  LR, NB, and SVM all perform comparably, consistent with the paper's finding in Section 4.3.1
3. **Random Forest does not dominate**-  model complexity yields no gains when the distributional signal between Show and Tell vocabulary is sufficiently clear
4. **Best overall: SVM** — LOPO AUC = 0.956
5. **CV inflated vs paper** — our dataset uses only the 130 one-shot sentences vs the paper's 351 (zero + one + two shot combined), causing higher fold overlap and higher CV estimates

---

## Notebook Contents

| Step | Description |
|---|---|
| 0 | Imports |
| 1 | Load & inspect dataset |
| 2 | Create binary label (Show=0, Tell=1) |
| 3 | Text pre-processing (5 steps) |
| 4 | Feature extraction + Distributional Hypothesis demo |
| 5 | Define classifiers |
| 6a | 5-Fold Cross-Validation |
| 6b | Leave-One-Plot-Out (LOPO) |
| 7 | Per-plot AUC bar chart |
| 8 | Figure 6 replication comparison chart |
| 9 | Full results summary table |
| 10 | Bonus: Random Forest vs paper classifiers |

---

## Installation & Usage

### Requirements

```bash
pip install numpy pandas matplotlib scikit-learn scipy
```

### Run

```bash
git clone https://github.com/ahmedzelia/show-and-tell-nlp.git
cd show-and-tell-nlp
jupyter notebook lab11_notebook.ipynb
```

The dataset is included in the repository. No path changes needed if you keep the files in the same directory.

---

## Paper Reference

```bibtex
@inproceedings{sivakumar2024showandtell,
  title     = {Show and Tell: Exploring Large Language Model's Potential in
               Formative Educational Assessment of Data Stories},
  author    = {Sivakumar, Naren and Chen, Lujie Karen and Papasani, Pravalika
               and Majmundar, Vigna and Feng, Jinjuan Heidi and Yarnall, Louise
               and Gong, Jiaqi},
  booktitle = {Gen4DS Workshop},
  year      = {2024},
  note      = {University of Maryland Baltimore County}
}
```

---

## Course Context

| Item | Details |
|---|---|
| Course | IS 733 — University of Maryland Baltimore County |
| Assignment | Lab 11 — NLP Classification of Data Stories |
| Dataset | `data_stories_one_shot.csv` (provided by course instructor) |
| Paper | Sivakumar et al., Gen4DS Workshop 2024 |

---

## Author

**Zeliatu Ahmed** — IS 733, UMBC
