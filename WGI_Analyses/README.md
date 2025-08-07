# GI Pain Project: Statistical Modeling and Analysis Codebase

This repository contains the full suite of code used for the analysis and modeling described in our manuscript. The focus is on understanding sex and gender differences in chronic pain through the generation of a weighted gender index.

---

## 1. System Requirements

### Operating System
- Linux, macOS, or Windows

### Python Version
- Python 3.9+

### Required Packages
Install the following Python libraries before running any scripts:

```bash
pip install pandas numpy statsmodels scikit-learn seaborn matplotlib scipy 
```

---

## 2. Installation Guide

1. **Clone the repository or download the code ZIP archive**:

```bash
git clone https://github.com/EVPlab/wgi-pain-analysis.git
cd wgi-pain-analysis
```

2. **(Optional but recommended) Create a virtual environment**:

```bash
python -m venv env
source env/bin/activate  # On Windows use: .\env\Scripts\activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

---
## 3. Mediation Analysis in R

### Overview

In addition to the Python-based analyses, an R script is included to perform parallelized mediation analysis using the `mediation` package on a set of binary outcome variables. This is used to evaluate the mediating role of a gender index variable in the relationship between sex and longitudinal pain outcomes.

### Required R Packages

Install the following R libraries:

```r
install.packages("mediation")
install.packages("foreach")
install.packages("doParallel")
```

---

## 4. Instructions for Use

### Scripts Overview

| Script | Purpose |
|--------|---------|
| Figure One |
| `Code_and_Split.py` | Preprocesses and splits UK Biobank data for modelling|
| `GI_nested_model.py` | Computes WGI |
| `GI_hormones.py` | Correlation Between WGI and Hormones |
| Figures Two and Three |
| `bodysite_sexdifference.py` | Analyzes odds ratios for different pain body sites by sex |
| `sex_diffs_dx.py` | Analyzes odds ratios for different pain diagnoses by sex |
| `UKB_OR.py` | Generates odds ratios for WGI relationship with pain variables from UK Biobank dataset |
| `nociplastic_epq_final.py` | Models nociplastic pain using EPQ variables |
| `OR_ave_nosex.py` | Aggregates odds ratios without sex stratification |
| `OR_ave_sexed.py` | Aggregates odds ratios with sex stratification |
| `Auto_Mediation.r` | Causal mediation analysis to invest WGI role in longitudinal sex differences in pain outcomes |
| Figure Five |
| `multilinearmodel_meno.py` | Multilinear Model predicting nociplastic pain conditions |

Typical run time for scripts are under 5 minutes
---

## 5. License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT), an Open Source Initiative-approved license.

---

## 6. Repository and Citation

> https://github.com/EVPlab/wgi-pain-analysis

If used in a publication, please cite as:

> Guglietti, G. et al. *Gender Norms as Sociocultural Determinants of Chronic Pain*. [Journal Name], [Year].

---

## 7. Further Information

See the **Methods** section of our manuscript for further details about our analyses or reach out to Gianluca via email (gianluca.guglietti@mail.mcgill.ca).
