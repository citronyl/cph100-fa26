# Diabetes Readmission Risk Stratification and Dataset Shift
#### CPH 100A Problem Set 3
#### Due Date: Fri Oct 30

## Introduction

Risk stratification identifies patients at elevated risk of outcomes such as readmission, new disease, or mortality. In this problem set, we will predict **readmission within 30 days** using ten years (1999–2008) of diabetes hospital encounters from 130 US hospitals and integrated delivery networks. You will develop conventional machine learning models, evaluate an LLM on structured clinical data, and investigate distribution shift between age groups.

This assignment is adapted from the *CPH 200C Computational Precision Health Cornerstone* risk stratification assignment (April 28, 2026). Some logistical details are adapted to this course and a free LLM API.

## Deliverables

There are no teams. Each student must submit their own **code** and **report**. You may discuss concepts with classmates, but your submission must be your own. 

**Code submission:** Completed model development and evaluation code, three LLM prompting variants, saved LLM responses/predictions (no API secrets), and plotting scripts.

**Report submission:** Point-by-point answers to Parts 1–3, plots, model comparisons, and a discussion of limitations. Submit via bCourses.

### Use of AI in this problem set

You may use LLMs for coding, debugging, conceptual help, and writing. We encourage you to attempt the core work yourself first and use AI to diagnose problems or improve your understanding. You must also evaluate an LLM directly as one of the models in Part 1.

## Part 0: Setup

### Installation

Install Python 3.10+ (Miniconda is recommended). From `psets/pset3`:

```bash
conda create -n cph100_pset3 python=3.10 -y
conda activate cph100_pset3
pip install -r requirements.txt
python check_installation.py
```

### Data access

Download **Diabetes 130-US Hospitals for Years 1999–2008** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008). Extract `diabetic_data.csv` and place it in `psets/pset3/data/diabetic_data.csv`. The dataset has one row per hospital encounter, with `readmitted` indicating `<30`, `>30`, or `NO`. Treat `<30` as the positive class and the other categories as negative.

Review the accompanying [Strack et al. (2014) paper](https://doi.org/10.1155/2014/781670) and the dataset documentation before modeling. Do not publish patient-level records or API credentials in your repository. The data are publicly distributed, but you should still minimize unnecessary disclosure of patient records to model providers.

### Code structure

- `data_utils.py`: Provided data loading and reproducible splitting utilities; inspect assumptions and adapt if needed.
- `models.py`: Provided preprocessing pipeline; **TODO** all three model families and hyperparameter search.
- `evaluation.py`: Provided bootstrap AUC confidence intervals; **TODO** demographic exploration, AUC, plots, and age-shift analysis.
- `llm_eval.py`: Provided Groq API transport/caching and response parsing; Provided structured and natural-language prompts; **TODO** alternative prompt format.
- `main.py`: Provided data-inspection/sample-generation entry point; **TODO** model-training orchestration.
- `check_installation.py`: Quick environment checks.

### Running the scaffold

```bash
python main.py --data data/diabetic_data.csv --inspect
python main.py --data data/diabetic_data.csv --make-llm-sample --sample-size 200
python llm_eval.py --sample outputs/llm_sample.csv --format structured --dry-run
```

The `--dry-run` command previews the two provided prompt formats. The `alternative` format requires you to implement its TODO first. To make live API calls, obtain a free key from [GroqCloud](https://console.groq.com), set `GROQ_API_KEY` in your environment (never put it in code), and remove `--dry-run`. The default model is `openai/gpt-oss-20b`. The [Groq free-plan limits](https://console.groq.com/docs/rate-limits) can change and include request **and token** quotas. For a sample of 200 encounters and three formats, expect up to 600 calls; you may need to spread calls over more than one day. You may instead use another accessible API model, documenting its name and limitations. The API code caches responses locally, so rerunning should not repeat completed requests.

## Part 1: Diabetes Risk Stratification and Dataset Shift 

### 1.1 Data Exploration

Graph the 30-day readmission rates by **age, gender, and race**. Which groups are associated with higher readmission rates? Reflect on sample sizes and how missingness affects interpretation.

### 1.2 Model Development

Use the provided preprocessing pipeline and implement classifiers from three model classes: **linear**, **tree-based**, and **neural network**. Implement hyperparameter selection on validation data, use multiple train/validation/test splits, and compare held-out performance; the original assignment suggests AUC in the neighborhood of **0.65–0.70**, though performance depends on preprocessing and splits.

In your report, describe your models, features, preprocessing decisions, hyperparameter search and selection, and the best overall model. Use the provided bootstrap confidence-interval helper and graph the models' **AUC with appropriate confidence intervals** (see Cortes and Mohri, 2004). Consider one-hot encoding categorical data, handling missing values, and interactions between features (see Strack et al., 2014).

Avoid test-set tuning. Because a patient may have more than one encounter, explain whether and how you prevent patient overlap across splits. `data_utils.py` provides encounter-level and patient-disjoint split utilities. Decide which design answers your question and assess potential leakage.

### 1.3 Large Language Model Evaluation

Select an LLM with programmatic API access; we suggest `openai/gpt-oss-20b` on the GroqCloud free tier. Evaluate the model on a **reproducible sample of 200 test encounters**, using the *same encounters* across three input formats:

1. A structured representation of patient fields.
2. A less-structured, natural-language representation.
3. Another modification of your choosing, such as abbreviations.

Use the provided structured-data and natural-language templates, and design a third prompt representation yourself. Ask the model to output a **probability between 0 and 1** of readmission within 30 days. Compare AUC across prompt formats and against trained models evaluated on that same sample. Record invalid responses and discuss what they imply. You may choose a smaller sample if quotas prevent completing 200; disclose the sample size, selection procedure and resulting uncertainty. Do not include the outcome `readmitted`, encounter ID or patient ID in model prompts.

### 1.4 Distribution Shift

The dataset does not include timestamps suitable for the proposed shift analysis. We will therefore simulate distribution shift using patient age.

Train your best **linear model** using encounters of patients **younger than 50**. Report its AUC on held-out encounters in that age group. Then evaluate the same trained model on patients **50 or older** (we use `>=50` so no one falls between groups). Compare your model's performance and explain plausible causes using differences in covariates, readmission rates or feature importance.

Compare the LLM's performance on patients under 50 versus 50 or older, using the same LLM predictions collected for Part 1.3. Comment on the uncertainty of subgroup estimates.

## Part 2: FDA Request for Comments

On January 7, 2025, the FDA issued [draft guidance on AI-enabled device software functions](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/artificial-intelligence-enabled-device-software-functions-lifecycle-management-and-marketing-submission). Read the draft guidance and describe **one or two notable aspects**.

Choose one public comment submitted in response to the FDA's request for comments, read it closely, and identify its author and position. What are they advocating, and how might their institutional role (industry, academia, clinical care or other sector) inform their position? The original assignment cited the April 7, 2025 comment deadline and 107 comments; those figures are historical and may no longer match the live docket. Include a link to the specific comment you read.

## Part 3: Feedback

This is a new problem set (compared to Problem Set 1 and 2). How long did this problem set take you, in hours? You may optionally describe your use of LLMs and suggestions for improving the assignment. 

## References

- Cortes, C., & Mohri, M. (2004). *Confidence intervals for the area under the ROC curve*. NeurIPS.
- Strack, B., et al. (2014). *Impact of HbA1c measurement on hospital readmission rates: Analysis of 70,000 clinical database patient records*. BioMed Research International.
- Nestor, B., et al. (2019). *Feature robustness in non-stationary health records*. MLHC.
- Kallus, N., & Zhou, A. (2019). *The fairness of risk scores beyond classification*. NeurIPS.
- Yao, H., et al. (2022). *Wild-Time: A benchmark of in-the-wild distribution shift over time*. NeurIPS.
