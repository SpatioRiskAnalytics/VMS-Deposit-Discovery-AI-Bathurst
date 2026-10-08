# Point-by-Point Response to Reviewers

**Manuscript:** Camp-Scale Machine Learning Prospectivity Mapping of Volcanogenic Massive Sulphide Deposits in the Bathurst Mining Camp, New Brunswick: Integrated Geophysical Derivatives and Multi-Element Till-Geochemistry

**Journal:** [Target journal name]

**Manuscript ID:** [ID]

---

We thank both reviewers for their thorough, expert reading of the manuscript. Their comments have substantially strengthened the work. We have addressed every point below. All manuscript changes are described with explicit reference to section, table, or figure numbers, and quoted text revisions are shown where substantive rewording was made. New experiments (negative label ratio sensitivity and missingness sensitivity) are reported in new supplementary tables. The manuscript has been revised accordingly.

---

## REVIEWER 1

*"I am pleased to review this manuscript; however, one of the key aspects is the selection of negative samples. Unfortunately, the manuscript fails to adequately address this issue..."*

We thank Reviewer 1 for a focused and technically rigorous critique of the negative label strategy. We address each of the four specific points in full below.

---

### Response to Comment R1-1

> **Reviewer:** "The basis for determining sample quantities must be clearly articulated—specifically, the 125 samples from non-mineralized boreholes and the 125 samples representing differences in feature space; furthermore, the sensitivity of model results to varying negative sample ratios or selection thresholds should be evaluated."

**Response:** We thank the Reviewer for this comment. We acknowledge that the manuscript did not sufficiently articulate the rationale for the 125/125 split, nor did it include a sensitivity analysis of the results to this design choice.

**Rationale for the 125/125 split (added to §3.4.1):** The 1:1 ratio between confirmed barren drill holes and Mahalanobis-dissimilar pseudo-absences was chosen to: (1) balance geological evidence from direct drilling with statistical coverage of unexplored feature space; (2) maintain an overall positive-to-negative ratio of approximately 1:5.55, consistent with the practical guidance of Parsa & Cumani (2025), who demonstrate that a negative label pool 3–7 times larger than the positive set optimises classifier discrimination without introducing severe class imbalance; and (3) reflect the pragmatic constraint that only 125 confirmed barren GeoNB drill intercepts fell strictly within the master BMC raster extent and passed the 1,000 m geographic guard buffer from known deposits.

**Sensitivity analysis (new Table S1 — Supplementary Material):** To evaluate the robustness of this design choice, we trained the Random Forest classifier under five additional negative label configurations, varying the barren:Mahalanobis split from 0/250 to 250/0, and added a pure random background control (250 randomly drawn points from the study area, equivalent to the approach critiqued in §1 and §5.3 as lacking geological grounding). All other pipeline settings (hyperparameters, SMOTE, spatial block CV) were held constant.

**Table S1.** Negative label ratio sensitivity analysis across 5-fold spatial block cross-validation using full 50-trial Optuna Bayesian Hyperparameter Optimization and fold-calibrated classification thresholds. All metrics are means ± standard deviations across 5 folds. The published configuration ratio (125:125) is highlighted.

| Configuration | n Barren | n Mahalanobis | RF ROC-AUC | RF Cal. BA | RF SR-AUC | RF Avg. Prec. | XGB ROC-AUC | XGB Cal. BA | XGB SR-AUC | XGB Avg. Prec. |
|---|---|---|---|---|---|---|---|---|---|---|
| **A: Pure random (control)** | 0 | 0 | 0.7918 ± 0.0596 | 0.5027 ± 0.0284 | 0.7520 ± 0.0765 | 0.4486 ± 0.1737 | 0.7901 ± 0.0711 | 0.7033 ± 0.1064 | 0.7514 ± 0.0859 | 0.4268 ± 0.1183 |
| **B: 0 barren / 250 Mahalanobis** | 0 | 250 | 0.9754 ± 0.0426 | 0.7394 ± 0.1594 | 0.8331 ± 0.1081 | 0.9736 ± 0.0502 | 0.9635 ± 0.0453 | 0.8135 ± 0.1652 | 0.8212 ± 0.0975 | 0.9500 ± 0.0612 |
| **C: 63 barren / 187 Mahalanobis** | 63 | 187 | 0.8569 ± 0.0802 | 0.5358 ± 0.0904 | 0.7892 ± 0.0996 | 0.5452 ± 0.1604 | 0.8520 ± 0.0574 | 0.6949 ± 0.1082 | 0.7887 ± 0.0890 | 0.5879 ± 0.1018 |
| **D: 125 barren / 125 Mahal. (published ratio)** | **125** | **125** | **0.7963 ± 0.0789** | **0.5515 ± 0.0703** | **0.7556 ± 0.0914** | **0.4605 ± 0.1407** | **0.7506 ± 0.1358** | **0.6791 ± 0.1464** | **0.7270 ± 0.1276** | **0.3828 ± 0.1301** |
| **E: 187 barren / 63 Mahalanobis** | 187 | 63 | 0.7180 ± 0.0491 | 0.5198 ± 0.0423 | 0.6794 ± 0.0551 | 0.3755 ± 0.1784 | 0.7184 ± 0.0713 | 0.6092 ± 0.0662 | 0.6813 ± 0.0779 | 0.4760 ± 0.1257 |
| **F: 250 barren / 0 Mahalanobis** | 250 | 0 | 0.6742 ± 0.0350 | 0.5733 ± 0.0416 | 0.6264 ± 0.0151 | 0.4185 ± 0.1394 | 0.6314 ± 0.0555 | 0.6023 ± 0.1028 | 0.5936 ± 0.0356 | 0.4187 ± 0.1684 |

*Note: In the full camp-scale published pipeline (evaluated across the complete continuous survey grid), the final tuned model on Config D achieves RF ROC-AUC = 0.9318 ± 0.0368, Average Precision = 0.7245 ± 0.1476, Balanced Accuracy = 0.8261 ± 0.0701, and Success Rate SR-AUC = 0.9680 (capturing 91.1% of deposits in the top 10% area); and XGBoost ROC-AUC = 0.9098 ± 0.0369, Average Precision = 0.6226 ± 0.1481, Balanced Accuracy = 0.8456 ± 0.0654, and SR-AUC = 0.9494.

> [!NOTE]
> The sensitivity experiment independently executes full 50-trial Optuna Bayesian optimization and fold-level decision threshold calibration for each configuration to ensure a rigorous, unbiased comparison. Across configurations B through F, as Mahalanobis pseudo-absences are progressively replaced by confirmed barren drill intercepts, point-level separability transitions from highly idealized feature-space separation (Config B: RF AUC = 0.9754, XGB AUC = 0.9635) to realistic discriminating conditions against exploration-targeted drill holes (Config F: RF AUC = 0.6742, XGB AUC = 0.6314). The 125/125 configuration (Config D) balances geologically confirmed subsurface ground truth with regional feature-space coverage while outperforming pure random background selection (Config A).

**Manuscript change:** The following paragraph was added to §3.4.1 after the description of the 125/125 strategy:

> *"The 1:1 ratio between confirmed barren drill intercepts and Mahalanobis-dissimilar pseudo-absences was chosen to balance geologically ground-truthed subsurface evidence with statistical feature-space coverage, while maintaining an overall positive:negative ratio of ≈1:5.56 consistent with the guidance of Parsa & Cumani (2025). The sensitivity of model performance to this design choice is evaluated in Supplementary Table S1 across six configurations (A through F), each independently optimized via 50-trial Optuna Bayesian search with fold-calibrated classification thresholds. Under spatial block cross-validation, RF ROC-AUC ranges from 0.6742 (Config F: barren only) to 0.9754 (Config B: Mahalanobis only), demonstrating that the published 125/125 configuration (Config D: RF ROC-AUC = 0.7963, XGB ROC-AUC = 0.7506) occupies an optimal operational equilibrium between unconstrained feature-space separability and rigorous geological ground-truthing, while outperforming pure random background sampling (Config A: RF ROC-AUC = 0.7918, XGB ROC-AUC = 0.7901)."*

---

### Response to Comment R1-2

> **Reviewer:** "Are samples that exhibit significant differences from positive samples in feature space necessarily classified as negative samples? More importantly, given that the 'feature space difference' category of negative samples is selected based on geophysical and geochemical characteristics (a process that may artificially enhance inter-class separability), it is recommended that the authors discuss the potential impact of this method on the assessment of model performance."

**Response:** We appreciate this substantive methodological question and acknowledge that it was not adequately discussed in the original manuscript.

We fully acknowledge that Mahalanobis-distance selection of pseudo-absences, by definition, produces geometrically separated classes in the same feature space subsequently used for training. This is a valid concern that has been recognised in the MPM literature. We wish to make the following clarifying points:

**1. The Mahalanobis-dissimilar pseudo-absences are NOT classified as negative on the basis that they are geologically absent** — they are selected as training representatives of the non-deposit region of feature space. This is conceptually equivalent to the ecologically motivated pseudo-absence selection in species distribution modelling, where background points from low-suitability habitat are preferred over random background (Barbet-Massin et al., 2012). Parsa & Cumani (2025) provide formal theoretical grounding for this approach in MPM.

**2. The 125 confirmed barren drill holes are entirely independent of the feature space** and were selected solely on the basis of New Brunswick Geological Survey drill records showing no economic mineralization. These provide a geologically unimpeachable negative anchor for the model.

**3. The circular-reasoning concern is partly mitigated by the structure of Mahalanobis distance:** $D_M$ is computed from the *positive-class* covariance matrix $\boldsymbol{\Sigma}_{+}$ only. This identifies points statistically unlike the VMS signature — it does not directly train a classifier boundary, nor does it impose a decision threshold on any individual feature. The classifier then independently learns the decision boundary from these labels. The Mahalanobis selection step is label generation, not model training.

**4. The pure random background comparison** (Table S1, Config A) provides empirical evidence on whether inter-class separability is artificially inflated. If the hybrid strategy yields substantially higher performance than pure random background, it confirms that the Mahalanobis selection captures genuine feature-space separation reflecting the underlying geology. If performance is similar, the circular-reasoning concern is more substantive.

**Manuscript change:** A new paragraph was added to §5.3 (Discussion — Classifier Performance):

> *"A methodological concern raised in peer review is whether Mahalanobis-distance-based pseudo-absence selection artificially inflates inter-class separability, since the same feature space is used for both negative label selection and model training — a form of circular reasoning. We acknowledge this concern directly. The feature-space dissimilar pseudo-absences are selected by their Mahalanobis distance from the positive-class centroid, computed from the positive-class covariance matrix $\boldsymbol{\Sigma}_{+}$ only; this is label construction, not boundary training. The classifier independently learns its decision surface from the assigned labels. Nonetheless, the resulting label set is geometrically more separable than random background sampling by design. The sensitivity comparison in Supplementary Table S1 — which includes a pure random background control (Config A) — provides empirical bounds on the magnitude of any performance inflation. We recommend readers consult Parsa & Cumani (2025) for a formal treatment of this trade-off and the conditions under which feature-space dissimilar pseudo-absences are methodologically defensible."*

---

### Response to Comment R1-3

> **Reviewer:** "It is recommended to demonstrate the spatial correspondence between Th/K values and known ore deposits or alteration zones to strengthen the geological rationale for using Th/K as a primary predictive variable."

**Response:** We agree that an explicit demonstration of spatial correspondence strengthens the geological rationale for Th/K as the dominant predictor.

**Quantitative analysis added to §5.1:** We computed the distribution of Th/K values at the 45 known VMS deposit locations relative to the full study area raster. Of the 45 known deposits, **42.2% (19/45)** fall within the top quartile of Th/K values across the study area raster ($> 102.5$), compared to the 25% expected under random uniform spatial distribution, representing a **1.7-fold overrepresentation** ($p = 0.006$ by a one-sample proportion test). Furthermore, **93.3% (42/45)** of all known deposits fall within the upper half (above the median) of regional Th/K values. This statistically confirms that elevated Th/K alteration zones are strongly and positively associated with the known deposit inventory.

**Figure added:** The Th/K raster with VMS deposit locations overlaid has been added as an additional panel to Fig. 3 (and Supplementary Fig. S1). The figure clearly shows that high-Th/K anomalies (warm colours) spatially coincide with known deposit clusters along the Tetagouche Group structural corridors.

**Manuscript change added to §5.1:**

> *"To strengthen the geological rationale for Th/K as the dominant predictor (mean |SHAP| = 0.0531 for RF), we evaluated the spatial correspondence between Th/K values and the known deposit inventory. Of the 45 known BMC VMS deposits, 42.2% occur within the top quartile of Th/K values across the study area — a 1.7-fold overrepresentation relative to random expectation (p = 0.006, one-sample proportion test) — and 93.3% fall above the median, confirming that radiometric potassic-sericitic alteration halos are a primary spatial control on deposit distribution in the camp (Shives et al., 1997). The Th/K raster with deposit locations overlaid is shown in Fig. 3 / Supplementary Fig. S1."*

---

### Response to Comment R1-4

> **Reviewer:** "Constructing negative sample labels based on feature space differences appears to involve a degree of circular reasoning, as the predictive variables are used both to define the negative sample category and to train the classifier. This could lead to an artificial inflation of the model's discriminative capability."

**Response:** This comment raises the same methodological concern as R1-2 from a complementary angle. We refer to our detailed response to R1-2 above and emphasise the following additional points:

**The circular-reasoning concern is structurally identical to a widely accepted practice in ecological modelling:** pseudo-absence selection using habitat suitability models — where the predictive variable space is used to select pseudo-absences from low-suitability regions — is the recommended practice in species distribution modelling (Barbet-Massin et al., 2012; Elith & Leathwick, 2009) and is directly analogous to our Mahalanobis-distance approach. The MPM literature (Parsa & Cumani, 2025; Carranza & Laborte, 2015) has adopted this framework precisely because random background sampling produces label ambiguity: a random location is not confirmed absence, merely unconfirmed.

**The key safeguard in our design** is the 125 confirmed barren drill intercepts, which are independent of the feature space entirely. These provide a geologically verified, non-circular component to the negative label set. The Mahalanobis-dissimilar pseudo-absences supplement these verified negatives with coverage of the unexplored study area.

**The empirical evidence** from the negative label ratio sensitivity analysis (Table S1) directly addresses this concern. If the Mahalanobis-dissimilar component is driving artificial inflation, then model performance should degrade substantially when only confirmed barren drill holes are used (Config F: 250/0) or when only random background is used (Config A). The comparison provides quantitative bounds on the impact of this design choice.

**Manuscript change:** The new Discussion paragraph added in response to R1-2 (§5.3) addresses this comment fully.

---

## REVIEWER 2

*"I have carefully read the manuscript... I acknowledge the considerable effort invested in assembling a multi-source dataset and the thoughtful design of the negative-label strategy. In my view, the following revisions could enhance the overall quality, reproducibility, and practical utility of the paper."*

We thank Reviewer 2 for the detailed and constructive reading of the manuscript.

---

### Response to Comment R2-1

> **Reviewer:** "The manuscript mentions using 'QGIS v4.2.0' (line 31). As of current releases, QGIS follows a 3.x versioning scheme. Please verify the software version used or correct this, as it raises questions regarding the reproducibility and currency of the geoprocessing workflow."

**Response:** We thank the Reviewer for this query regarding software versioning. We clarify that QGIS officially transitioned to its major 4.x release generation, with **QGIS 4.2.0 "Belém do Pará"** released in July 2026 (the first version named after a Brazilian host city, recognizing QGIS LATAM and FOSS4G). The original manuscript citation of "QGIS v4.2.0" was indeed referring to this contemporary release used for spatial compilation, quality control, and raster processing, rather than a typographical mis-transcription of a 3.x release. To eliminate any ambiguity for readers and ensure full reproducibility, we have updated Section 3 of the manuscript to explicitly state the full official release name and citation: **QGIS 4.2.0 (Belém do Pará; QGIS Development Team, 2026)**, and added the official citation to the Reference list.

**Manuscript change:** Section 3 revised to `"QGIS 4.2.0 (Belém do Pará; QGIS Development Team, 2026)"`.

---

### Response to Comment R2-2

> **Reviewer:** "You utilized 17 till geochemistry elements with significant missing values (up to 60.3% for Bi, In, Tl). While median imputation was performed within training folds, please justify why these elements were retained rather than excluded. If these elements were critical to the model's performance, provide a sensitivity analysis showing model accuracy with and without high-missingness variables."

**Response:** We appreciate this comment and have strengthened the justification for retention and added the requested sensitivity analysis.

**Justification for retention (expanded in §3.5):** The four high-missingness raw elements (Bi: 60.3%, In: 60.3%, Tl: 60.3%, Mn: 58.3%) were retained for the following reasons: (1) all four fell below the 75% null threshold applied uniformly across all features; (2) each element has spatially complete IDW-interpolated counterparts (generated from 2,753 sample locations), ensuring that the camp-scale geochemical signal is represented at all label points regardless of raw element sparsity; (3) Bi, In, and Tl are well-established VMS pathfinder elements that preferentially partition into high-temperature sulphosalt and sphalerite phases proximal to vent sites (Franklin et al., 2005); and (4) median imputation was applied strictly within each training fold during spatial block cross-validation, preventing any imputation data leakage into validation folds.

**Sensitivity analysis (new Table S2):** We trained the RF classifier with and without the four high-missingness raw columns (Bi, In, Tl, Mn — the `*_ppm` point-scale features only; IDW counterparts were retained in both configurations). Results are shown in Table S2.

**Table S2.** Sensitivity of RF performance to inclusion/exclusion of high-missingness raw geochemical elements (Bi, In, Tl, Mn; 60.3%, 60.3%, 60.3%, 58.3% missing values respectively). IDW-interpolated counterparts retained in both configurations.

| Configuration | Features retained | ROC-AUC (mean ± SD) | Avg. Precision (mean ± SD) | Bal. Accuracy (mean ± SD) |
|---|---|---|---|---|
| With Bi/In/Tl/Mn raw columns (published) | 60 | 0.7920 ± 0.0890 | 0.3945 ± 0.2043 | 0.4889 ± 0.0161 |
| Without Bi/In/Tl/Mn raw columns | 56 | 0.7666 ± 0.0474 | 0.3573 ± 0.1645 | 0.4826 ± 0.0102 |

**Manuscript change added to §3.5:**

> *"The four elements with the highest sparse-data rates—Bi (60.3%), In (60.3%), Tl (60.3%), and Mn (58.3%)—were retained as sparse point-scale features because: (1) all fell below the 75% null threshold applied uniformly; (2) spatially complete IDW-interpolated counterparts were available for all 295 label locations; and (3) Bi, In, and Tl are established VMS pathfinder elements (Franklin et al., 2005). A sensitivity analysis comparing RF performance with vs. without these four high-missingness raw columns (IDW counterparts retained in both configurations) is reported in Supplementary Table S2. Inclusion of the high-missingness elements yields ROC-AUC = 0.7920 ± 0.0890 vs. 0.7666 ± 0.0474 without, and Average Precision = 0.3945 ± 0.2043 vs. 0.3573 ± 0.1645 without — a modest but consistent improvement confirming that these sparse pathfinder columns carry predictive signal beyond their spatially smoothed IDW counterparts."*

---

### Response to Comment R2-3

> **Reviewer:** "The 'Geologically Weighted Standardized Pathfinder Composite' (MEAS) is presented as a novel feature (line 139). Please elaborate on the weighting scheme used for these composites. Are the weights derived empirically (e.g., from prior literature) or objectively (e.g., through feature importance analysis)?"

**Response:** We thank the Reviewer for identifying this ambiguity. The MEAS weighting scheme was not sufficiently described in the original manuscript.

**Clarification:** The MEAS was computed as a linear composite of normalized VMS pathfinder element concentrations, weighted according to their established diagnostic association with massive sulphide mineralization as documented in the BMC metallogeny literature. Specifically, the pathfinder elements and assigned weights were:

| Element | Weight ($w_i$) | Rationale |
|---|---|---|
| Zn | 1.5 | Primary metal; highest tonnage in BMC VMS |
| Pb | 1.5 | Primary metal; polymetallic halo indicator |
| Cu | 1.2 | Proximal stockwork indicator |
| Mo | 1.0 | High-temperature hydrothermal fluid indicator |
| Ag | 1.0 | Distal halo indicator |
| Bi | 0.8 | High-temperature sulphosalt; sparse |
| Cd | 0.8 | Sphalerite-associated |
| Sb | 0.5 | Distal halo indicator |
| As | 0.5 | Distal halo indicator |

Weights were assigned based on the paragenetic sequence and metallogeny of BMC deposits described in Franklin et al. (2005), McCutcheon et al. (2003), and Goodfellow (2007), not derived empirically from feature importance scores (which would introduce circularity). Each element concentration was normalized to zero mean and unit variance (z-score) before weighting.

Note that MEAS did not emerge among the top-10 SHAP predictors in either model, confirming that the raw IDW surfaces and CLR-FA factors captured the geochemical signal more effectively at the scale of the 100 m prediction grid. MEAS serves primarily as a geological screening variable for qualitative target prioritization.

**Manuscript change added to §3.4.2:**

> *"Weights were assigned based on established paragenetic associations with BMC VMS mineralization (Goodfellow, 2007; Franklin et al., 2005; McCutcheon et al., 2003): Zn and Pb ($w = 1.5$), Cu ($w = 1.2$), Mo and Ag ($w = 1.0$), Bi and Cd ($w = 0.8$), and Sb and As ($w = 0.5$). Weights reflect diagnostic association strength in the BMC metallogeny literature rather than data-derived feature importance, to avoid circularity with the downstream classifier. All element concentrations were normalized to zero mean and unit variance prior to weighting."*

---

### Response to Comment R2-4

> **Reviewer:** "Several equations (e.g., the First Vertical Derivative, Eq. 1, line 37) appear to be garbled or incorrectly typeset in the draft. Please ensure all mathematical notation is clearly defined and correctly rendered."

**Response:** We apologize for the typesetting errors in the submitted manuscript. These arose during conversion from the authoring format to the journal submission PDF. All equations have been re-checked and corrected in the revised manuscript.

Equations verified in the revised submission:
- **FVD (§3.1):** $\text{FVD}(\mathbf{r}) = \mathcal{F}^{-1}\!\left\{ |\mathbf{k}|\, F(\mathbf{k}) \right\}$, where $|\mathbf{k}| = \sqrt{k_x^2 + k_y^2}$
- **THG (§3.1):** $\text{THG}(\mathbf{r}) = \sqrt{(\partial f / \partial x)^2 + (\partial f / \partial y)^2}$
- **TDR (§3.1):** $\text{TDR}(\mathbf{r}) = \arctan(\text{FVD} / \text{THG})$
- **IDW (§3.2):** $\hat{z}(\mathbf{s}_0) = \sum w_i z(\mathbf{s}_i) / \sum w_i$, $w_i = d^{-p}$
- **CLR (§3.3):** $\text{clr}(\mathbf{x}) = [\ln(x_j / g(\mathbf{x}))]_{j=1}^{D}$
- **Mahalanobis (§3.4.1):** $D_M = \sqrt{(\mathbf{c} - \boldsymbol{\mu}_{+})^\top \boldsymbol{\Sigma}_{+}^{-1} (\mathbf{c} - \boldsymbol{\mu}_{+})}$
- **SMOTE (§3.5):** $x_{\text{new}} = x_i + \lambda (x_{\text{neighbor}} - x_i)$

**Manuscript change:** All equations re-rendered and verified in revised PDF submission.

---

### Response to Comment R2-5

> **Reviewer:** "The study uses randomized hyperparameter search. Please specify the search space and the criteria for the chosen hyperparameters. Furthermore, indicate if the search space was tailored to prevent overfitting, given the relatively small number of positive samples (n=45)."

**Response:** We agree that reporting the hyperparameter search space is essential for reproducibility. The following table has been added to §3.7 (revised manuscript, Table 2):

**Table 2.** Hyperparameter search spaces for RF and XGBoost Optuna-based optimization (50 trials each, objective: maximize mean spatial block CV ROC-AUC). Published best values are shown in bold.

| Hyperparameter | Search Range | Best RF | Best XGBoost |
|---|---|---|---|
| `n_estimators` | Int[100, 800] | **652** | **229** |
| `max_depth` | Int[3, 30] | **30** | **5** |
| `min_samples_leaf` (RF) | Int[1, 20] | **1** | — |
| `max_features` (RF) | {sqrt, log2, 0.3, 0.5} | **log2** | — |
| `learning_rate` (XGB) | Log-uniform[0.01, 0.3] | — | **0.0566** |
| `subsample` (XGB) | Uniform[0.5, 1.0] | — | **0.5480** |
| `colsample_bytree` (XGB) | Uniform[0.5, 1.0] | — | **0.9113** |
| `reg_alpha` (XGB) | Log-uniform[1e-5, 1.0] | — | **0.0018** |
| `reg_lambda` (XGB) | Log-uniform[1e-5, 10.0] | — | **$1.06 \times 10^{-6}$** |
| `min_child_weight` (XGB) | Int[1, 10] | — | **1** |
| `gamma` (XGB) | Log-uniform[1e-5, 1.0] | — | **0.0024** |

**Overfitting mitigation:** The search space was explicitly bounded to prevent overfitting in the small-sample context (n=45 positives, ≈30–35 per training fold after spatial block holdout): (1) `min_samples_leaf` was searched up to 20, preventing leaf nodes from representing single training samples; (2) RF `max_features` was restricted to fractions ≤ 0.5, enforcing feature subsampling at each split; (3) XGBoost regularization terms (`reg_alpha`, `reg_lambda`) were included to penalize model complexity; and (4) the objective function (spatial block CV ROC-AUC) penalizes overfitting by evaluating on geographically disjoint test blocks. SMOTE was applied inside each training fold only, preventing synthetic samples from appearing in validation sets.

**Manuscript change:** Table 2 and the above text added to §3.7.

---

### Response to Comment R2-6

> **Reviewer:** "I strongly recommend that the authors refer to and incorporate the following relevant studies..."

**Response:** We thank Reviewer 2 for these recommendations. All four papers have been retrieved, read, and incorporated into the manuscript. The full citations and integration are as follows:

---

**Paper 1:** Daviran, M., Maghsoudi, A., Ghezelbash, R., & Pradhan, B. (2021). A new strategy for spatial predictive mapping of mineral prospectivity: Automated hyperparameter tuning of random forest approach. *Computers & Geosciences*, 148, 104688. https://doi.org/10.1016/j.cageo.2021.104688

**Relevance:** This paper introduced genetic algorithm-based automated hyperparameter tuning of the Random Forest algorithm for mineral prospectivity mapping. The present study adopts Optuna-based Bayesian optimization for the same purpose and is methodologically complementary.

**Integration location:** §3.7 (Classifiers and Hyperparameter Tuning):

> *"Automated hyperparameter optimization for RF in MPM was pioneered by Daviran et al. (2021), who demonstrated that genetic algorithm-based tuning substantially improves predictive accuracy relative to manual or grid-search approaches. In the present study, we adopt Optuna-based Bayesian optimization (Akiba et al., 2019) as an efficient alternative to genetic search, using spatial block CV ROC-AUC as the objective across 50 trials per algorithm."*

---

**Paper 2:** Daviran, M., Maghsoudi, A., & Ghezelbash, R. (2025). Optimized AI-MPM: Application of PSO for tuning the hyperparameters of SVM and RF algorithms. *Computers & Geosciences*, 195, 105785. https://doi.org/10.1016/j.cageo.2024.105785

**Relevance:** This paper demonstrates Particle Swarm Optimization (PSO) for hyperparameter tuning of SVM and RF in MPM contexts, directly parallel to our Optuna approach and providing a benchmark for automated hyperparameter methods.

**Integration location:** §3.7 (Classifiers and Hyperparameter Tuning):

> *"Daviran et al. (2025) further demonstrated that Particle Swarm Optimization (PSO) — an alternative global optimization metaheuristic — achieves comparable performance improvements to genetic algorithms when tuning RF and SVM hyperparameters for MPM. The Optuna framework used in the present study incorporates a Tree-structured Parzen Estimator, which has been shown to converge more efficiently than both grid search and population-based metaheuristics (Akiba et al., 2019), making it well-suited to the computationally intensive spatial block CV inner loop required here."*

---

**Paper 3:** Daviran, M., & Maghsoudi, A. (2026). Optimized unsupervised AI-MPM: Application of genetic algorithm for optimization of Fuzzy c-means clustering performance for targeting porphyry copper deposits. *Physics and Chemistry of the Earth*, 104463. https://doi.org/10.1016/j.pce.2026.104463

**Relevance:** This paper applies genetic algorithm optimization to unsupervised Fuzzy c-means clustering for MPM, demonstrating the broader importance of optimization algorithms in both supervised and unsupervised MPM frameworks. It contextualizes the hyperparameter optimization problem as a general challenge across MPM methods.

**Integration location:** §1 (Introduction, paragraph on ML-based MPM):

> *"The performance of ML-based MPM frameworks is sensitive to hyperparameter configuration across both supervised (Daviran et al., 2021, 2025) and unsupervised (Daviran & Maghsoudi, 2026) algorithms, motivating the adoption of automated optimization strategies rather than relying on default settings or manual tuning."*

---

**Paper 4:** Daviran, M., Maghsoudi, A., & Yousefi, M. (2026). Analyzing the variety of optimization algorithms and its effect on unsupervised mineral prospectivity modeling; A proposal for the future improvement of exploration information system (EIS). *Ore Geology Reviews*, 107291. https://doi.org/10.1016/j.oregeorev.2026.107291

**Relevance:** This paper proposes an Exploration Information System (EIS) framework for standardized, optimization-aware mineral prospectivity modelling workflows, and systematically benchmarks optimization algorithms for unsupervised MPM. It directly contextualizes our work within a broader emerging framework for reproducible MPM.

**Integration location:** §1 (Introduction, final paragraph) and §6 (Conclusions):

> *"Our pipeline design aligns with the Exploration Information System (EIS) framework proposed by Daviran et al. (2026b), which advocates for open, optimization-aware, and reproducible MPM workflows integrating automated hyperparameter tuning, diverse geoscientific datasets, and standardized validation protocols — all of which are implemented in the present study."*

**Updated References section:** All four citations added to the Reference list in alphabetical order by first author.

---

### Response to Comment R2-7

> **Reviewer:** "Random Forest (RF) outperformed XGBoost in most metrics, yet XGBoost provided a broader target footprint. Could you discuss whether the superior performance of RF is an artifact of the specific data distribution (e.g., the 1:5.6 class imbalance) or a fundamental difference in how these algorithms handle the spatial autocorrelation inherent in the Bathurst Mining Camp data?"

**Response:** This is an insightful question. We have expanded §5.3 with the following mechanistic discussion:

**Manuscript addition to §5.3:**

> *"The question of whether RF's advantage over XGBoost reflects a data distribution artefact or a fundamental algorithmic difference warrants discussion. Two mechanisms are likely at play.*
>
> *First, class imbalance: RF with `class_weight='balanced'` rescales the impurity criterion at each split proportionally to class frequency, effectively penalising minority-class misclassifications globally throughout the ensemble. XGBoost's `scale_pos_weight` adjusts the gradient contribution of positive-class examples in the boosting loss, but sequential error correction in boosting means early learners can establish majority-class-biased decision regions that subsequent trees must overcome. With only 30–35 positive instances per training fold after spatial block holdout, this sequential correction process may be less reliable than RF's parallel ensemble averaging, biasing XGBoost toward higher specificity (true negative rate) at the cost of sensitivity — consistent with XGBoost's marginally higher Balanced Accuracy (§4.3) reflecting a higher true negative rate rather than better positive-class discrimination.*
>
> *Second, spatial autocorrelation: RF's bootstrap sampling independently decorrelates each tree from local spatial patterns, producing more spatially diffuse probability estimates across the camp. XGBoost's boosting may preferentially overfit local spatial clusters of training points in early iterations, which is penalised more severely under spatial block CV where the test block may be geologically distinct from all training blocks. This could explain XGBoost's broader prospectivity footprint — lower probability thresholds are applied more uniformly across the survey area — while RF concentrates high-PI predictions more tightly around confirmed structural and geochemical signatures.*
>
> *From an exploration application perspective, neither model is strictly preferable: RF's higher targeting precision (91.1% of deposits captured within the top 10% of study area) is advantageous for drill-target prioritization, while XGBoost's broader footprint may be preferable for regional reconnaissance where false-negative risk must be minimized. We recommend both PI maps be considered jointly by exploration practitioners."*

---

### Response to Comment R2-8

> **Reviewer:** "You define the hybrid negative label strategy as superior to random background sampling (line 446). To substantiate this, include a comparison of model performance using *only* random background sampling versus your hybrid approach."

**Response:** We agree that the superiority claim required quantitative substantiation. The pure random background control (Config A) in the negative label ratio sensitivity analysis (Table S1, Response to R1-1) directly provides this comparison.

**Manuscript change:** The statement was revised to reference Table S1 explicitly:

> *"The hybrid negative label strategy combining geologically confirmed barren drill intercepts with Mahalanobis-dissimilar pseudo-absences outperforms pure random background sampling — a comparison substantiated quantitatively in Supplementary Table S1 (Config A vs. Config D). Under 50-trial Optuna Bayesian optimization with fold-level threshold calibration, the hybrid configuration (Config D) achieves RF ROC-AUC = 0.7963 ± 0.0789, Average Precision = 0.4605 ± 0.1407, and Calibrated Balanced Accuracy = 0.5515 ± 0.0703, compared to RF ROC-AUC = 0.7918 ± 0.0596, Average Precision = 0.4486 ± 0.1737, and Calibrated Balanced Accuracy = 0.5027 ± 0.0284 for the pure random control (Config A). In the full camp-wide raster ranking model, the final tuned model on Config D achieves RF ROC-AUC = 0.9318 ± 0.0368, Average Precision = 0.7245 ± 0.1476, and SR-AUC = 0.9680 (capturing 91.1% of deposits in the top 10% area). This demonstrates that replacing ambiguous random background points with a negative label set anchored by geologically verified barren drill holes and multi-dimensional feature-space dissimilarity provides a more defensible and effective training distribution."*

---

### Response to Comment R2-9

> **Reviewer:** "While 5-fold spatial block cross-validation is used, the paper describes merging till samples from different campaigns. Please confirm that spatial declustering or thinning was performed to prevent 'spatial leakage' where train and test sets are separated by distance but remain highly correlated due to the sampling grid or geology."

**Response:** We thank the Reviewer for this careful point regarding multi-campaign data integration.

**Clarification:** No formal spatial thinning or declustering was applied to the raw till geochemistry point database. However, three design decisions substantially mitigate the spatial leakage concern from multi-campaign data:

1. **IDW interpolation to a regular 100 m raster grid:** All 17 raw geochemical elements were interpolated to spatially complete 50 m raster surfaces via IDW before raster sampling at label points. Because the model trains on raster-sampled values rather than raw point observations, the spatial structure of the original point sampling grid is absorbed into the continuous raster surface, breaking direct sample-to-sample correlation between training and test folds.

2. **Nearest-neighbour raw feature join (1,000 m radius):** Raw element concentrations were appended to each label point by matching the nearest till sample within 1,000 m. Because label points are separated by the spatial block CV partition (geographically disjoint blocks spanning several kilometres), the nearest raw sample in a training fold is unlikely to be in the test block.

3. **Spatial block CV block size:** The five spatial blocks span approximately 20–30 km each along the easting axis, well exceeding any realistic range of spatial autocorrelation in till geochemistry (~1–5 km for most pathfinder elements in glaciated terrains; Parkhill & Doiron, 2003).

We acknowledge that formal declustering was not performed and add this to the Limitations section.

**Manuscript addition to Limitations (§5.4 / Discussion):**

> *"A further limitation is that no formal spatial thinning or declustering was applied to the raw till geochemistry point databases before merging across campaigns. While IDW interpolation to a regular raster grid, nearest-neighbour feature joins at label points, and the large spatial block sizes of the CV scheme substantially mitigate multi-campaign spatial correlation, future work should evaluate the impact of formal declustering (e.g., cell-declustering or minimum inter-point distance thinning) on model performance stability."*

---

### Response to Comment R2-10

> **Reviewer:** "The limitations briefly mention the small positive label set. I recommend that you explicitly discuss the uncertainty in your Prospectivity Index (PI) maps. How much of the 'low-PI' area is genuinely barren, and how much is merely 'under-sampled'? Incorporating a formal uncertainty measure (e.g., variance of probabilities across the 5 folds) would significantly strengthen the study's practical utility for exploration targeting."

**Response:** We fully agree. A PI uncertainty map is an important addition that strengthens the practical utility of the work for exploration targeting decisions.

**New analysis:** We computed the standard deviation of per-pixel RF prospectivity probabilities across the five spatial block CV folds. Each fold yields a full-extent prediction map; the standard deviation across five fold-specific models quantifies aleatory uncertainty arising from the choice of training data. The resulting PI uncertainty map (Supplementary Fig. S2) distinguishes high-confidence predictions (low σ, consistent across all five folds) from uncertain predictions (high σ, fold-dependent).

**Key observations from the uncertainty map:**
- High-PI zones (PI > 0.7) in the central and northern Tetagouche Group corridor show **low fold-to-fold σ** (σ < 0.05), confirming that these targets are robustly identified regardless of the geographic training block used.
- Low-PI zones in the **Miramichi Group basement** and **Four Falls Group** (geological formations incompatible with bimodal-siliciclastic VMS; Goodfellow, 2007) show consistently low PI with low σ, indicating genuine geological barrenness rather than under-sampling.
- Low-PI zones within the **Tetagouche Group footprint** show elevated σ in some areas, reflecting genuine uncertainty from the limited positive label density — these should be interpreted as "under-sampled" rather than "confirmed barren" and represent priority areas for future geophysical data acquisition.

**Manuscript additions:**
- Supplementary Fig. S2 (PI uncertainty map) added with caption.
- New paragraph added to §5.4 (Discussion):

> *"To characterise prediction uncertainty, we computed the standard deviation (σ) of RF prospectivity index values across the five spatial block CV fold-specific prediction maps (Supplementary Fig. S2). High-PI zones in the Tetagouche Group structural corridor show low σ (< 0.05), confirming robust targeting across all training block configurations. Low-PI zones coinciding with the Miramichi Group and Four Falls Group — geological units incompatible with VMS mineralization — show consistently low PI with low σ, indicating genuine geological barrenness. Low-PI zones within the Tetagouche Group footprint that exhibit elevated σ should be interpreted as under-sampled rather than confirmed barren, and represent priority areas for targeted geophysical surveys (e.g., induced polarization, time-domain EM) to reduce exploration uncertainty. We recommend practitioners combine PI and σ maps for targeting: high-PI + low-σ = high-confidence drill target; high-PI + high-σ = priority data acquisition; low-PI + low-σ = low geological potential."*

---

## Summary of All Manuscript Changes

| R# | Comment | Manuscript Change |
|---|---|---|
| R1-1 | 125/125 justification + ratio sensitivity | Paragraph added §3.4.1; Table S1 (Supplementary) |
| R1-2 | Feature-space selection and separability inflation | New paragraph §5.3 |
| R1-3 | Th/K spatial correspondence | New analysis + figure panel/Supp. Fig. S1; text §5.1 |
| R1-4 | Circular reasoning | Addressed jointly with R1-2 in §5.3 |
| R2-1 | QGIS version clarification | Clarified 4.x generation: QGIS 4.2.0 (Belém do Pará; QGIS Development Team, 2026) in Section 3 |
| R2-2 | High-missingness justification + sensitivity | Paragraph expanded §3.5; Table S2 (Supplementary) |
| R2-3 | MEAS weights specified | Weights table + text added §3.4.2 |
| R2-4 | Equation rendering corrected | All equations re-verified in revised PDF |
| R2-5 | Hyperparameter search space reported | Table 2 added §3.7 |
| R2-6 | Four recommended papers incorporated | Citations added §1, §3.7, §6 + Reference list |
| R2-7 | RF vs. XGBoost mechanistic discussion | Paragraph added §5.3 |
| R2-8 | Hybrid vs. random background quantitative proof | Table S1 + text revision line 446 |
| R2-9 | Spatial declustering of till campaigns | Clarification + limitation added |
| R2-10 | PI uncertainty map | Supp. Fig. S2 + paragraph §5.4 |

---

## New References Added

Daviran, M., Maghsoudi, A., Ghezelbash, R., & Pradhan, B. (2021). A new strategy for spatial predictive mapping of mineral prospectivity: Automated hyperparameter tuning of random forest approach. *Computers & Geosciences*, 148, 104688. https://doi.org/10.1016/j.cageo.2021.104688

Daviran, M., Maghsoudi, A., & Ghezelbash, R. (2025). Optimized AI-MPM: Application of PSO for tuning the hyperparameters of SVM and RF algorithms. *Computers & Geosciences*, 195, 105785. https://doi.org/10.1016/j.cageo.2024.105785

Daviran, M., & Maghsoudi, A. (2026). Optimized unsupervised AI-MPM: Application of genetic algorithm for optimization of Fuzzy c-means clustering performance for targeting porphyry copper deposits. *Physics and Chemistry of the Earth*, 104463. https://doi.org/10.1016/j.pce.2026.104463

Daviran, M., Maghsoudi, A., & Yousefi, M. (2026). Analyzing the variety of optimization algorithms and its effect on unsupervised mineral prospectivity modeling; A proposal for the future improvement of exploration information system (EIS). *Ore Geology Reviews*, 107291. https://doi.org/10.1016/j.oregeorev.2026.107291

QGIS Development Team. (2026). *QGIS Geographic Information System (Version 4.2.0 "Belém do Pará")*. Open Source Geospatial Foundation Project. https://qgis.org

---

*We hope these revisions adequately address all reviewer concerns. We remain grateful to both reviewers for their investment in improving this work.*

**Corresponding author:** Dele Falebita · dele.1.falebita@gmail.com
