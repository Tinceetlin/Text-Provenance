# Black-Box Detection of Z.AI-Generated and Modified Academic Abstracts Against Human-Written Controls: Discrimination, Calibration, and Repeatability of GLM-5.2 Verbalized Confidence

**Black-Box Detection of Z.AI-Generated and Modified Academic Abstracts Against Human-Written Controls: Discrimination, Calibration, and Repeatability of GLM-5.2 Verbalized Confidence**

This repository accompanies the paper above. It contains the prompts, the data, and the statistical analyses used to test how well **GLM-5.2 (Z.AI)** can tell human-written academic abstracts apart from abstracts that it generated, rewrote, or paraphrased, using only black-box API access (no weights, hidden activations, or logits).

> **Responsible use.** Provenance scores from an LLM are decision-support evidence for screening, not proof of authorship. The results describe one model, one prompt, and one API configuration, and should not be used as definitive evidence that a person used AI.

## Overview

| | |
|---|---|
| **Model under test** | GLM-5.2 (Z.AI), API model identifier `glm-5.2` |
| **Task** | Classify an abstract as `non-Z.AI` (human control) or `Z.AI` (generated or modified by Z.AI) |
| **Source corpus** | 1,000 English Computer Science abstracts archived on or before 30 September 2021, from the [arXiv Paper Abstracts](https://www.kaggle.com/datasets/spsayakpaul/arxiv-paper-abstracts) dataset (Kaggle) |
| **Test set** | 4,000 texts: 1,000 human controls plus 1,000 Generate, 1,000 Rewrite, and 1,000 Paraphrase variants |
| **Judgments** | 4,000 requests, each eliciting 5 judgments, giving 20,000 judgment-level outputs |
| **Outputs per judgment** | Predicted class and verbalized confidence scores for both classes |

The study asks three questions:

1. **RQ1.** How well can GLM-5.2 separate human-written abstracts from Z.AI-generated, rewritten, and paraphrased ones?
2. **RQ2.** How good are its verbalized confidence scores (ROC-AUC, Brier Score, calibration)?
3. **RQ3.** How stable are its scores and labels across five judgments of the same text?

## Key results

All values come from the per-text mean of five Z.AI-class confidence scores, thresholded at 0.5 (an exact 0.5 is assigned to Z.AI). Each scenario pairs the same 1,000 human controls with 1,000 variants. Values in parentheses are 95% percentile CIs from 10,000 source-family bootstrap replicates.

| Metric | Control vs Paraphrase | Control vs Rewrite | Control vs Generate |
|---|---|---|---|
| ROC-AUC | 0.9266 (0.9155–0.9372) | 0.9472 (0.9372–0.9566) | **0.9952** (0.9926–0.9973) |
| Accuracy | 85.60% (84.20–87.00) | 89.10% (87.80–90.35) | **97.55%** (96.85–98.20) |
| Precision (Z.AI) | 95.41% | 95.78% | 96.48% |
| Recall (Z.AI) | 74.80% | 81.80% | 98.70% |
| F1-score (Z.AI) | 83.86% | 88.24% | 97.58% |
| Brier Score (lower is better) | 0.1147 (0.1066–0.1230) | 0.0920 (0.0850–0.0993) | **0.0354** (0.0319–0.0390) |
| ECE, 10 bins (lower is better) | 0.0828 (0.0742–0.0948) | **0.0781** (0.0685–0.0904) | 0.1019 (0.0955–0.1087) |
| False-negative rate | 25.20% | 18.20% | 1.30% |
| False-positive rate | 3.60% | 3.60% | 3.60% |

**What stands out**

- **Detection weakens as the transformation gets subtler.** Accuracy falls from Generate (97.55%) to Rewrite (89.10%) to Paraphrase (85.60%). The differences are significant (Cochran's Q = 248.17, df = 2, p < 0.001), and all three McNemar–Holm pairwise comparisons are significant. One in four paraphrased abstracts (25.20%) was missed.
- **Good discrimination is not good calibration.** Generate has the best ROC-AUC and Brier Score but the highest ECE. Rewrite and Paraphrase do not differ significantly in ECE (ΔECE = −0.0047, Holm-adjusted p = 0.548), and their ordering flips across 5, 10, 15, and 20 bins.
- **Stable is not the same as correct.** 222 of 252 missed Paraphrase texts (88.10%) received the same wrong label in all five judgments. Unanimity should not be read as correctness.
- **Human false positives are consistent.** The same 36 human controls (3.60%) were flagged as Z.AI in every scenario.

### Repeatability across five judgments

| Group | Texts | Mean Z.AI-class score | Median between-judgment SD (pp) | Full 5:0 agreement | Mean label entropy (bits) |
|---|---|---|---|---|---|
| Non-Z.AI control | 1,000 | 16.82% | 2.14 | 95.90% | 0.03682 |
| Z.AI-Generate | 1,000 | 89.82% | 1.71 | 98.20% | 0.01648 |
| Z.AI-Rewrite | 1,000 | 73.01% | 2.58 | 92.60% | 0.06587 |
| Z.AI-Paraphrase | 1,000 | 67.74% | 2.70 | 90.10% | 0.08915 |

Across all 4,000 texts, 3,768 had 5:0 agreement, 68 had a 4:1 split, and 164 had a 3:2 split.

### 1. Data selection

Records from the arXiv Paper Abstracts dataset were filtered to English-language Computer Science abstracts archived on or before 30 September 2021 (54,230 eligible records). They were processed in dataset order, with no random sampling, until 1,000 complete source families were retained. Source abstracts were excluded and replaced by the next eligible record when a transformation or classification response failed validation (152 in total).

### 2. Abstract transformation

Each human abstract (the control) was converted into three variants by GLM-5.2:

| Variant | What the model did |
|---|---|
| **Z.AI-Generate** | Wrote a new abstract from the original **title only** |
| **Z.AI-Rewrite** | Comprehensively rewrote the abstract in a different style while preserving the core content |
| **Z.AI-Paraphrase** | Altered wording and phrasing to mask the human author's stylistic signatures |

### 3. Scenarios

The same 1,000 controls are paired with each variant set, giving three binary scenarios of 2,000 texts each (4,000 unique texts in total). Because the controls are reused, the three scenarios are not independent control samples. Each scenario is 50% positive, whereas the full collection is 75% Z.AI variants.

### 4. Context-isolated repeated-judgment protocol

Each abstract is sent in a fresh API request containing only the classification prompt and that abstract, with no conversation history. One response contains **five judgments** (not five separate API calls), each with a predicted class and two verbalized confidence scores that sum to 1. Responses are validated against the required JSON structure.

```json
{
  "run": 1,
  "prediction": "non-Z.AI or Z.AI",
  "confidence_non-zai": 0.0000,
  "confidence_zai": 0.0000,
  "reason": "<maximum 300 words>"
}
```

The five Z.AI-class scores are averaged into one **Mean Verbalized Confidence Score** per text. A mean of 0.5 or higher gives a `Z.AI` prediction. Z.AI is the positive class.

### 5. Model configuration

| Parameter | Value |
|---|---|
| Endpoint | `https://api.z.ai/api/paas/v4/chat/completions` |
| API model identifier | `glm-5.2` |
| Temperature | `0.0` |
| Thinking | Enabled |
| Reasoning effort | Max |
| Call period | 22 July to 16 August 2026 |

No immutable model snapshot or build identifier was recorded, so results reflect the hosted service during that period.

### 6. Metrics and statistics

- **Classification:** Accuracy, Precision, Recall, F1-score, with Z.AI as the positive class.
- **Discrimination and probability quality:** ROC-AUC and Brier Score on the per-text mean score.
- **Calibration:** Expected Calibration Error with 10 equal-width bins (sensitivity checked with 5, 15, and 20 bins) and reliability diagrams.
- **Repeatability:** Between-judgment score standard deviation, label entropy, and label-agreement patterns (5:0, 4:1, 3:2).
- **Inference:**
  - 10,000 source-family bootstrap replicates for 95% CIs.
  - Paired randomization tests on ECE differences, Holm-adjusted.
  - Cochran's Q, then McNemar tests with Yates' correction and Holm adjustment, for accuracy across conditions.
  - Friedman tests for confidence scores and between-judgment SD.

## Repository structure

```text
Text-Provenance/
├── Prompt_engineering/   # Prompts for Generate, Rewrite, Paraphrase, and provenance classification
├── dataset/              # Human control abstracts, Z.AI variants, and classification results
├── statistical_test/     # Statistical analyses (calibration, bootstrap, Cochran's Q, McNemar, Friedman)
├── figures/              # Figures shown in this README
└── README.md
```

Two identifiers link the records: `source_id` ties each human control to its Generate, Rewrite, and Paraphrase variants, and `text_id` groups the five judgments made for one text.

## Limitations

- **One model, one setup.** The study covers GLM-5.2 on English Computer Science abstracts under one prompt and one API configuration. Results should not be generalized without re-evaluation.
- **No other-LLM controls.** The negative class contains only human-written text. The results cannot separate Z.AI-specific self-recognition from sensitivity to generic LLM-style cues.
- **Possible training-data overlap.** The September 2021 cutoff defines eligibility but does not guarantee independence from the model's training data.
- **Prevalence and control reuse.** Scenario-level metrics assume 50% prevalence, and the controls are shared across scenarios.
- **Replacement of failed records.** The 152 replaced records were substituted sequentially, not randomly, and the effect on corpus composition was not quantified.
- **Limited repeatability view.** Five judgments within one response do not capture long-term stability under future service changes.
- **No same-data baseline.** No alternative detector or LLM judge was run on the same 4,000 texts, so comparisons with prior work are contextual, not a controlled ranking.

## Data and attribution

- Source abstracts: S. Paul and S. Rakshit, [arXiv Paper Abstracts](https://www.kaggle.com/datasets/spsayakpaul/arxiv-paper-abstracts), Kaggle, 2021. Please follow the terms of the original dataset when reusing the human-written abstracts.
- Model documentation: [GLM-5.2, Z.AI Developer Documentation](https://docs.z.ai/guides/llm/glm-5.2).

## Citation

If you use this repository, please cite it as:

```bibtex
@misc{sir2026,
  author       = {Sir, Yosua Alberth and Tallo, Tince Etlin},
  title        = {{Text-Provenance}},
  year         = {2026},
  version      = {1.0.0},
  howpublished = {GitHub},
  url          = {https://github.com/Tinceetlin/Text-Provenance}
}
```

The accompanying paper is *Black-Box Detection of Z.AI-Generated and Modified Academic Abstracts Against Human-Written Controls: Discrimination, Calibration, and Repeatability of GLM-5.2 Verbalized Confidence* (Y. A. Sir and T. E. Tallo). Volume, issue, and DOI will be added once published.

## Authors

- **Yosua Alberth Sir**: Department of Information Technology Education, Universitas Nusa Cendana, Kupang, Indonesia. [yosuasir@staf.undana.ac.id](mailto:yosuasir@staf.undana.ac.id)
- **Tince Etlin Tallo**: Department of Computer Science, Universitas Nusa Cendana, Kupang, Indonesia. [tince.tallo@staf.undana.ac.id](mailto:tince.tallo@staf.undana.ac.id)
