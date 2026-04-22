# Sleep Apnea Detection — Tableau Dashboard Specification

**Dashboard title:** Sleep Apnea Detection — Model Evaluation Dashboard
**Layout:** 2×2 grid, 1600×900 px
**Data sources:** Four Hyper extracts in `tableau/`

---

## Data Sources

| Hyper File | Tableau Connection Name | Used In |
|---|---|---|
| `record_inventory.hyper` | `RecordInventory` | Sheet 1 |
| `train_features.hyper` | `TrainFeatures` | Sheet 2 |
| `test_predictions.hyper` | `TestPredictions` | Sheets 3, 4 |
| `episode_stats.hyper` | `EpisodeStats` | (supplemental) |

**Dataset note:** PhysioNet Apnea-ECG (Penzel et al., 2000). 70 overnight ECG recordings, 100 Hz sampling rate, per-minute apnea/normal annotations.

**Model note:** XGBoost classifier trained on 13 HRV + episode-context features. Subject-level 5-fold GroupKFold cross-validation.

---

## Calculated Fields (define before building sheets)

```
// In RecordInventory
Severity Color
IF [apnea_fraction] < 0.10 THEN '#27ae60'
ELSEIF [apnea_fraction] < 0.30 THEN '#f1c40f'
ELSE '#e74c3c' END

// In TestPredictions
Minute Accuracy Band
IF [correct] THEN 1 ELSE 0 END

Per-Subject Accuracy
SUM([correct]) / COUNT([record_id])

Accuracy Tier
IF [Per-Subject Accuracy] > 0.85 THEN 'High (>85%)'
ELSEIF [Per-Subject Accuracy] > 0.70 THEN 'Medium (70–85%)'
ELSE 'Low (<70%)' END

Apnea Ground Truth Band
IF [true_label] = 'A' THEN 0.999 ELSE NULL END
```

---

## Sheet 1 — Apnea Burden by Subject

**Data source:** `RecordInventory`
**Chart type:** Vertical bar chart

| Field | Shelf | Notes |
|---|---|---|
| `record_id` | Columns | Sort descending by apnea_burden |
| `apnea_burden` (×100) | Rows | Create calc: `[apnea_fraction]*100` |
| `severity_category` | Color | Traffic-light palette (see below) |
| `record_id`, `group`, `n_apnea_minutes`, `severity_category` | Tooltip | Shown on hover |

**Color palette:**
- Mild (< 10%) → `#27ae60` (green)
- Moderate (10–30%) → `#f1c40f` (yellow)
- Severe (> 30%) → `#e74c3c` (red)
- Unknown → `#bdc3c7` (grey)

**Axis:** Label Y-axis "Apnea Burden (%)"
**Title:** "Apnea Burden by Subject"
**Reference lines:** At 10%, 30%, 50% (dashed, labelled)

---

## Sheet 2 — HRV Feature Space by Apnea Label

**Data source:** `TrainFeatures`
**Chart type:** Scatter plot

| Field | Shelf | Notes |
|---|---|---|
| `lf_hf_ratio` | Columns | X-axis |
| `rmssd` | Rows | Y-axis |
| `label` | Color | A=#F44336 (red), N=#2196F3 (blue) |
| `record_id` | Detail | For subject-level filter |
| `record_id` | Filter | Show as dropdown quick-filter |

**Mark opacity:** 50%
**Mark size:** 1–2 (smallest)
**Title:** "HRV Feature Space by Apnea Label"
**Subtitle:** "Each point = one minute window. Filter by record to explore individual subjects."

---

## Sheet 3 — Per-Subject Prediction Accuracy

**Data source:** `TestPredictions`
**Chart type:** Horizontal bar chart

| Field | Shelf | Notes |
|---|---|---|
| `Per-Subject Accuracy` (calc) | Columns | Format as % |
| `record_id` | Rows | Sort descending by accuracy |
| `Accuracy Tier` (calc) | Color | High=#27ae60, Medium=#f1c40f, Low=#e74c3c |

**Reference line:** At overall test set accuracy (constant line, label "Overall")
**Title:** "Per-Subject Prediction Accuracy"
**Tooltip:** `record_id`, accuracy %, true apnea fraction, predicted apnea fraction

---

## Sheet 4 — Predicted Apnea Probability Timeline

**Data source:** `TestPredictions`
**Chart type:** Dual-axis line + area

**Primary axis (line):**
| Field | Shelf |
|---|---|
| `minute_index` | Columns |
| `predicted_proba` | Rows (primary axis) |

**Secondary axis (area shading):**
- Create a dual axis with `Apnea Ground Truth Band` (calc field)
- Set mark type to **Area**, color = #F44336, opacity = 20%
- Synchronize axes (both 0–1 range)
- Hide secondary axis header

**Filter:** `record_id` → dropdown quick-filter
**Title:** "Predicted Apnea Probability vs Ground Truth"
**Subtitle:** "Red shading = ground-truth apnea minutes. Line = model-predicted probability."
**Threshold reference line:** At 0.5 (dashed, label "Decision threshold")

---

## Dashboard Assembly

1. **Canvas size:** Fixed, 1600 × 900 px
2. **Layout containers:** Use a vertical container holding two horizontal containers
   ```
   [Vertical Container — full width]
     [Title Banner — text object, full width, ~60 px height]
     [Horizontal Container — top row, ~50% height]
       [Sheet 1 — left half]
       [Sheet 2 — right half]
     [Horizontal Container — bottom row, ~50% height]
       [Sheet 3 — left half]
       [Sheet 4 — right half]
     [Footer — text object, full width, ~40 px height]
   ```
3. **Title banner text:**
   `Sleep Apnea Detection — Model Evaluation Dashboard`
   Font: 18 pt bold, centred, background: #2c3e50, color: white
4. **Footer text box:**
   ```
   Dataset: PhysioNet Apnea-ECG Database (Penzel et al., 2000)
   Model: XGBoost classifier | Features: 13 HRV + episode-context | CV: Subject-level 5-fold GroupKFold
   ```
5. **Filter action:** Sheet 3 → Sheet 4 on `record_id` (clicking a subject bar filters the timeline)
6. **Highlight action:** Sheet 2 → Sheet 2 on `label` (clicking a color legend entry highlights that class)

---

## Export Checklist

- [ ] All four Hyper files load without errors
- [ ] Sheet 1 shows 70 bars (35 train + 35 test) — or filter to one group
- [ ] Sheet 2 scatter shows both red and blue points across all records
- [ ] Sheet 3 shows 35 test records with accuracy bars
- [ ] Sheet 4 timeline responds to record_id filter
- [ ] Dashboard title banner visible
- [ ] Filter action (Sheet 3 → Sheet 4) tested with at least 3 records
- [ ] Workbook saved as `tableau/SleepApnea_Dashboard.twbx`
