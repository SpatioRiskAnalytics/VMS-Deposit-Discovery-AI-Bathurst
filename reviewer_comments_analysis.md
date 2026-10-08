# Reviewer Comments — Dissection & Response Strategy
**Manuscript:** Camp-Scale Machine Learning Prospectivity Mapping of VMS Deposits in the Bathurst Mining Camp

---

## Overview

Both reviewers engage seriously with the work. **Reviewer 1** focuses almost entirely on the **negative label strategy**, raising concerns about circular reasoning and information leakage. **Reviewer 2** is broader and more technical, covering software versioning, missing data, equations, hyperparameters, literature, model comparison, validation, and uncertainty. The tone of Reviewer 2 is constructive and expert-level.

The good news: none of the comments require re-running the full pipeline. Most are addressable through **additional text, tables, figures, and one or two supplementary analyses**.

---

## REVIEWER 1 — Comment-by-Comment Dissection

### R1-C1: Basis for the 125/125 split and sensitivity to negative sample ratios

**What they're actually asking:**
- Why exactly 125 from each negative label source?
- What happens if the ratio of barren drill holes to Mahalanobis-dissimilar points changes (e.g., 200/50, 50/200)?
- What happens if the Mahalanobis distance threshold changes?

**Manuscript status:** The manuscript justifies the 1:1 split qualitatively (operational balance between geological evidence and feature-space coverage) but provides no quantitative sensitivity analysis.

**Suggested response:**
1. **Explain the 125/125 choice explicitly** in §3.4.1: it was chosen to (a) keep the 1:1 barren-to-feature-dissimilar ratio as a neutral starting point and (b) maintain an overall positive:negative ratio of 1:5.56, which is consistent with published guidance (Parsa & Cumani, 2025). Add a sentence or short paragraph.
2. **Run a sensitivity analysis** varying the split (e.g., 0/250, 63/187, 125/125, 187/63, 250/0) and report ROC-AUC and SR-AUC for each. A small table or supplementary figure is sufficient. This directly addresses the leakage concern by showing results are stable across configurations.
3. Also vary the Mahalanobis guard distance (500 m, 1 km, 2 km) to show the model is not sensitive to the exact cutoff.

> [!IMPORTANT]
> This is the highest-priority revision. If results are stable across ratios, it substantially refutes the leakage concern.

---

### R1-C2: Are feature-space dissimilar points necessarily true negatives? And does this inflate inter-class separability?

**What they're actually asking:**
This is the most philosophically challenging comment. The reviewer argues that selecting points that are maximally "unlike" deposits in the feature space used for training could:
- Make the classes artificially more separable (by construction)
- Mean that the negative label doesn't reflect true geological absence

**Manuscript status:** §3.4.1 and §5.3 partially address this but do not directly confront the circular-reasoning concern.

**Suggested response:**
1. **Acknowledge the concern explicitly in Discussion §5.3 or a new §5.5** ("Limitations and Potential Biases of the Hybrid Negative Label Strategy"): state that feature-space dissimilarity selection *does* produce geometrically separable classes by construction, and that this is both a strength (clear decision boundary) and a limitation (performance metrics may reflect the label strategy, not purely geological signal).
2. **Distinguish between two interpretations:** (a) The reviewer's concern about "artificial inflation" is valid in a closed system, but (b) the Mahalanobis-dissimilar points were selected *from the master BMC raster grid*, meaning they must exist within the known geological footprint of the camp. This constrains them geologically, even if not through direct drilling.
3. **Reference Parsa & Cumani (2025)** more prominently here — this paper is specifically about this methodological trade-off and provides theoretical backing.
4. **Add the pure-random-background comparison** (see R2-C8 below), which will empirically show whether performance inflation is occurring.

---

### R1-C3: Demonstrate spatial correspondence between Th/K and known deposits/alteration zones

**What they're actually asking:**
The reviewer wants a figure or table showing that high Th/K values spatially coincide with known VMS deposits and alteration zones, strengthening the geological rationale for using Th/K as the top predictor.

**Manuscript status:** §5.1 discusses Th/K dominance and its geological basis (Shives et al., 1997) but there is no figure showing this spatial correspondence explicitly.

**Suggested response:**
1. **Add a panel to Fig. 3 or create Fig. 11** showing the Th/K raster with VMS deposit locations overlaid. A simple visual of coincidence between high-PI anomalies and known deposits is compelling.
2. **Add a short quantitative statement**: e.g., "Of the 45 known VMS deposits, X% occur within the top quartile of Th/K values across the study area, consistent with the hydrothermal potassic-sericitic alteration model of Shives et al. (1997)."
3. If SHAP partial dependence plots (PDPs) are already computed, show the PDP for Th/K — the direction and shape of the curve directly demonstrate the nature of the relationship.

---

### R1-C4: Circular reasoning — using predictive variables to define negative labels AND to train the classifier

**What they're actually asking:**
This is R1-C2 restated more formally. The concern is that the same features used to select Mahalanobis-dissimilar negatives are also the features used to train the model. This could mean:
- The classifier is, in part, trained to discriminate on the very axis that was used to select the negatives
- This is a form of label leakage

**Manuscript status:** Not directly addressed.

**Suggested response:**
1. **Directly acknowledge this in the Discussion**: "A valid methodological concern is that Mahalanobis distance-based negative label selection uses the same feature space as the downstream classifier, introducing a form of circular reasoning."
2. **Counter-argument (strong):** Mahalanobis distance is computed from the *positive class covariance matrix only* — it identifies points that are statistically unlike the VMS deposit signature. It does NOT train or tune a classifier boundary. The classifier then learns the boundary independently from these labels. This is structurally similar to stratified sampling, not data leakage.
3. **The empirical counter:** The sensitivity analysis (R1-C1) and the pure-random comparison (R2-C8) provide quantitative evidence. If model performance degrades only modestly when using purely random negatives, it argues the circular reasoning concern is real. If performance is similar, it argues the Mahalanobis selection is essentially capturing a well-grounded geologically dissimilar set.
4. **Conceptual analogy**: This is equivalent to selecting ecological "pseudo-absences" in regions with no habitat suitability — a widely accepted practice that necessarily uses the predictor space. Cite Barbet-Massin et al. (2012) here.

> [!NOTE]
> R1-C2 and R1-C4 are the same concern stated twice. One comprehensive response in the Discussion covering both will suffice.

---

## REVIEWER 2 — Comment-by-Comment Dissection

### R2-C1: QGIS version listed as "v4.2.0" — incorrect versioning

**What they're actually asking:** QGIS uses a 3.x versioning scheme. "v4.2.0" does not exist and raises reproducibility concerns.

**Manuscript status:** A typo — line 31 of the manuscript references "QGIS v4.2.0".

**Suggested response:**
- **Verify the actual QGIS version used** and correct the citation. Current stable release is QGIS 3.38 (as of 2026). If the actual version was 3.4.2, it may have been mis-transcribed as "4.2.0".
- A one-line correction in the Methods. No further action needed.

> [!TIP]
> Check your project files or conda/pip environment logs for the exact version used.

---

### R2-C2: High-missingness elements (Bi, In, Tl — 60.3%) retained without justification; sensitivity analysis requested

**What they're actually asking:**
- Why keep elements with >60% missing values?
- Do these elements actually contribute to model performance, or are they noise?

**Manuscript status:** §3.4.2 and §3.5 explain retention (below the 75% null threshold; IDW counterparts available) but do not provide a sensitivity analysis showing their contribution.

**Suggested response:**
1. **Justify retention explicitly** (already partially done): IDW-interpolated counterparts are spatially complete; sparse raw values preserve deposit-proximal high-contrast anomalies. Bi is a known VMS pathfinder (Franklin et al., 2005).
2. **Add a sensitivity run**: Report model ROC-AUC and SR-AUC with vs. without the four high-missingness elements (Bi, In, Tl, Mn). If performance drops, it proves their value. If it doesn't change, acknowledge their marginal contribution and note they were retained for geochemical completeness. A 2-row table in supplementary material is sufficient.
3. **Note median imputation within training folds** is already described — this directly prevents leakage. Emphasise this more clearly.

---

### R2-C3: MEAS weighting scheme not explained — are weights empirical or from literature?

**What they're actually asking:**
The MEAS formula (Eq. in §3.4.2) uses weights $w_i$ but the manuscript doesn't state where these weights come from.

**Manuscript status:** §3.4.2 states weights are "according to their diagnostic association with massive sulphide mineralization" — vague.

**Suggested response:**
1. **Specify the weights explicitly** in §3.4.2: list the pathfinder elements included and their assigned weights. State clearly whether weights are:
   - Literature-derived (e.g., from VMS geochemical footprint studies: Franklin et al., 2005; McCutcheon et al., 2003)
   - Empirically derived from SHAP or feature importance
   - Equal weights (simplest — defensible if so)
2. A small table: Element | Weight | Source/Rationale would resolve this entirely.
3. If MEAS was a minor contributor to the final model (it doesn't appear in the top-10 SHAP features), note this and that its primary function was as a geological screening tool rather than a top classifier feature.

---

### R2-C4: Equations appear garbled (e.g., FVD equation, Eq. 1)

**What they're actually asking:**
Mathematical notation in the manuscript appears incorrectly typeset — likely a LaTeX rendering issue in the submitted PDF.

**Manuscript status:** The equations look correct in the `.md` source but may render incorrectly in the journal submission format.

**Suggested response:**
1. **Verify all equations render correctly in the final submission PDF.** Re-render and check especially:
   - FVD equation (Eq. 1): $\text{FVD}(\mathbf{r}) = \mathcal{F}^{-1}\!\left\{ |\mathbf{k}|\, F(\mathbf{k}) \right\}$
   - THG, TDR, IDW, CLR, SMOTE, Mahalanobis equations
2. Ensure all variables are **defined on first appearance** with a legend or notation table if needed.
3. Check the PDF directly — if the journal uses a different typesetting pipeline, some LaTeX commands may not render.

> [!WARNING]
> This is a formatting issue, not a scientific one — but it damages reproducibility perception significantly. Fix before resubmission.

---

### R2-C5: Hyperparameter search space not specified; no discussion of overfitting prevention

**What they're actually asking:**
- What ranges did the Optuna search cover for each hyperparameter?
- Were the search space bounds set to prevent overfitting (e.g., limiting tree depth, minimum samples per leaf)?

**Manuscript status:** §3.7 mentions Optuna-based randomized search but provides no search space details.

**Suggested response:**
1. **Add a Table 2 (Hyperparameter Search Spaces)** listing:
   - RF: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features — with the ranges searched
   - XGBoost: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, reg_alpha, reg_lambda — with ranges
2. State the **number of Optuna trials** run and the objective metric (e.g., mean CV ROC-AUC across 5 folds).
3. Explicitly note overfitting mitigations: min_samples_leaf ≥ 2 or 3 for RF (given only ~30-35 positive instances per training fold); XGBoost regularization terms (reg_alpha, reg_lambda > 0); early stopping if used.

---

### R2-C6: Recommend incorporating four specific papers into Introduction/Methodology

**The four papers:**
1. https://doi.org/10.1016/j.cageo.2021.104688
2. https://doi.org/10.1016/j.cageo.2024.105785
3. https://doi.org/10.1016/j.pce.2026.104463
4. https://doi.org/10.1016/j.oregeorev.2026.107291

**Suggested response:**
1. **Retrieve and read these papers** (check citation audit summary in workspace for any already reviewed).
2. **Integrate citations contextually** — do not just append them. Find the most natural home for each:
   - MPM methodology papers → Introduction §1.4 or §3.7
   - Negative label / pseudo-absence papers → §3.4.1
   - ML in mineral exploration → §1.3
3. Write 1–2 sentences per paper explaining how each relates to and contextualizes your work.

> [!IMPORTANT]
> Reviewer 2 explicitly recommended these — ignoring them would likely result in rejection. Retrieve and incorporate all four before resubmission.

---

### R2-C7: RF outperformed XGBoost — is this due to class imbalance or spatial autocorrelation?

**What they're actually asking:**
A genuinely interesting scientific question: does RF's advantage stem from (a) how it handles the 1:5.6 class imbalance, or (b) intrinsic differences in how RF vs. XGBoost handle spatially autocorrelated features?

**Manuscript status:** §5.3 compares performance but doesn't discuss this mechanistically.

**Suggested response:**
Add a paragraph in §5.3 addressing:
1. **Class imbalance:** RF with `class_weight='balanced'` adjusts leaf probabilities globally, while XGBoost's `scale_pos_weight` adjusts the gradient loss — both handle imbalance, but RF's ensemble averaging may produce better-calibrated probability outputs for small positive classes.
2. **Spatial autocorrelation:** RF's bootstrap sampling already provides some internal spatial decorrelation; XGBoost's sequential boosting may overfit local spatial patterns in early iterations, which is penalized more severely under spatial block CV where the test block may be geologically distinct.
3. **The broader footprint of XGBoost:** Note that XGBoost's slightly lower BA but broader target footprint may be preferable for exploration applications where *recall* (capturing all deposits) is prioritized over *precision*. Frame this as a practical exploration trade-off, not a model failure.

---

### R2-C8: Compare hybrid negative labels vs. pure random background sampling quantitatively

**What they're actually asking:**
The manuscript *claims* the hybrid strategy is superior (line 446) but provides no quantitative comparison.

**Manuscript status:** No such comparison exists in the manuscript.

**Suggested response:**
1. **Run an additional experiment**: Train the same RF and XGBoost models with 250 randomly selected background points as negatives (instead of the hybrid set). Report ROC-AUC, AP, BA, and SR-AUC.
2. **Present as Table 3** (or supplementary): "Comparison of hybrid vs. random background negative label strategies."
3. This experiment simultaneously addresses R1-C4 (circular reasoning) — if the hybrid strategy truly outperforms, it validates the approach; if performance is similar, acknowledge this and soften the "superiority" claim.

> [!IMPORTANT]
> This is a **critical experiment** that directly responds to both Reviewer 1's circular reasoning concern and Reviewer 2's quantitative proof request. It should be prioritized.

---

### R2-C9: Spatial declustering — were till samples from different campaigns thinned to prevent spatial leakage?

**What they're actually asking:**
When merging till data from multiple campaigns, nearby samples from the same grid may end up in different CV folds but remain highly correlated — a subtle form of spatial leakage.

**Manuscript status:** §3.6 describes spatial block CV and §4.1.2 describes the coordinate-rounding merge, but spatial declustering is not mentioned.

**Suggested response:**
1. **Clarify what coordinate rounding did**: rounding to the nearest meter creates unique locations but does NOT thin spatially clustered samples from the same campaign.
2. **State whether declustering was performed**: if not, acknowledge this as a limitation.
3. **Mitigating factor to emphasize**: because the IDW surfaces are interpolated to a 50 m grid before sampling, the spatial correlation of raw point samples is absorbed into the raster — the model trains on raster pixels, not raw points. This substantially reduces (though doesn't eliminate) the campaign-clustering concern.
4. If raw elements are used as point features (via nearest-neighbor join), note that the 1,000 m search radius means each label point gets one raw sample, reducing the risk of correlated samples being in both folds.

---

### R2-C10: Uncertainty in PI maps — formally quantify and discuss under-sampled low-PI areas

**What they're actually asking:**
- How much of the "low-PI" area is genuinely barren vs. just under-sampled?
- Report fold-level variance of PI as a formal uncertainty measure.

**Manuscript status:** §6 (Conclusions) and §4.5 report PI statistics but no uncertainty map is presented. Limitations (lines 447–450) mention the small positive label set.

**Suggested response:**
1. **Compute and map fold-level PI variance**: For each raster pixel, compute the standard deviation of the 5-fold PI predictions. Generate a spatial uncertainty map (high σ → high uncertainty).
2. **Add this as Fig. 11 or supplementary**: a PI map alongside a PI uncertainty map is a powerful combined output.
3. **Expand the Limitations/Discussion section**: Acknowledge that low-PI zones in geologically untested areas (Miramichi Group basement, Four Falls Group) are confidently barren *geologically*, while low-PI zones in the Tetagouche Group footprint may reflect under-sampling rather than absence.
4. **Frame as exploration guidance**: high-PI + low-uncertainty = high-confidence drill target; high-PI + high-uncertainty = priority for further data acquisition.

> [!TIP]
> The fold-level PI predictions are already computed during CV — extracting variance across the 5 folds requires only a few lines of code and no re-training.

---

## Summary Action Table

| # | Reviewer | Comment | Action | Effort |
|---|---|---|---|---|
| R1-C1 | R1 | Justify 125/125 split; sensitivity to ratios | Write text + sensitivity analysis table | Medium |
| R1-C2 | R1 | Feature-space negatives inflate separability | Expand Discussion §5.x; cite Parsa & Cumani | Low |
| R1-C3 | R1 | Spatial correspondence of Th/K and deposits | Add figure panel or table with quantification | Low |
| R1-C4 | R1 | Circular reasoning concern | Address in Discussion alongside R1-C2 | Low |
| R2-C1 | R2 | QGIS version error | Correct one line | Trivial |
| R2-C2 | R2 | High-missingness elements retention | Justify + sensitivity run (with/without) | Medium |
| R2-C3 | R2 | MEAS weights not specified | Add weights table to §3.4.2 | Low |
| R2-C4 | R2 | Equation typesetting garbled | Re-check PDF rendering of all equations | Low |
| R2-C5 | R2 | Hyperparameter search space not reported | Add hyperparameter table | Low |
| R2-C6 | R2 | Four specific papers to incorporate | Read, cite, and integrate all four | Medium |
| R2-C7 | R2 | RF vs. XGBoost mechanistic comparison | Expand §5.3 with mechanistic discussion | Low |
| R2-C8 | R2 | Quantitative comparison: hybrid vs. random negatives | **Run new experiment**, add comparison table | High |
| R2-C9 | R2 | Spatial declustering of multi-campaign till data | Clarify in text; address as limitation | Low |
| R2-C10 | R2 | PI uncertainty quantification | Compute fold-variance map; expand Discussion | Medium |

---

## Key Experiments to Run

The following new analyses are required to fully respond:

1. **Negative label sensitivity** (R1-C1 + R2-C8):
   - Vary barren:Mahalanobis split: (0/250), (63/187), (125/125), (187/63), (250/0)
   - Also add: (250 pure random background points)
   - Report RF ROC-AUC, AP, SR-AUC for each
   - ~6 training runs (fast)

2. **High-missingness element sensitivity** (R2-C2):
   - Train with all 17 elements vs. without {Bi, In, Tl, Mn}
   - Report RF ROC-AUC and SR-AUC
   - ~2 training runs

3. **PI uncertainty map** (R2-C10):
   - Extract per-pixel predictions from each of the 5 CV folds
   - Compute standard deviation across folds per pixel
   - Export as GeoTIFF and add to manuscript

4. **Th/K spatial correspondence** (R1-C3):
   - Overlay VMS deposit locations on Th/K raster
   - Compute % of deposits within top quartile of Th/K values

---

## Strongest Assets to Leverage in Revision

- The **spatial block CV** already addresses the most common criticism in data-driven MPM.
- The **SMOTE-within-fold** implementation is methodologically clean and well-described.
- **SHAP interpretability** provides geological transparency that reviewers appreciate.
- **Parsa & Cumani (2025)** is directly cited and provides strong methodological backing for the Mahalanobis negative label approach.
- The **confirmed barren drill holes** component of the hybrid strategy is geologically unimpeachable — emphasise this as a key distinction from purely data-driven pseudo-absence methods.
