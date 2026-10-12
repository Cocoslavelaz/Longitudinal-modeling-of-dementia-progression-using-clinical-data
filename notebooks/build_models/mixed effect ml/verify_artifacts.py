"""Independently audit persisted outputs against the source CSV and original split recipe."""
import hashlib
import json
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

here = Path(__file__).resolve().parent
root = here.parents[2]
out = root / 'results/merf_nan_CU_AD_nonAD'
original = root / 'notebooks/build_models/ml/next_mmse_svm_rf_xgboost_CU_AD_nonAD.ipynb'
assert hashlib.sha256(original.read_bytes()).hexdigest() == (here / 'original_notebook.sha256').read_text().strip()
status = json.loads((out / 'execution_status.json').read_text(encoding='utf-8'))
assert status['real_data'] == 'completed'
notebook = nbformat.read(here / 'next_mmse_merf_nan_CU_AD_nonAD.ipynb', as_version=4)
nbformat.validate(notebook)
for c in notebook.cells:
    if c.cell_type == 'code':
        assert c.execution_count is not None
        assert all(o.output_type != 'error' for o in c.outputs)

raw = pd.read_csv(root / 'data/processed/model_df_next_mmse3.csv')
raw['visit_date'] = pd.to_datetime(raw.visit_date)
raw = raw.sort_values(['patient_id', 'visit_date'])
audit = json.loads((out / 'data_audit.json').read_text(encoding='utf-8'))
pred = pd.read_csv(out / 'test_predictions.csv', parse_dates=['history_end_date', 'target_date'])
assert not pred.duplicated(['Group', 'patient_id']).any()
assert (pred.history_end_date < pred.target_date).all()
assert np.isfinite(pred[['true_mmse', 'rf_prediction', 'merf_population_prediction', 'merf_calibrated_prediction', 'b_hat']]).all().all()
np.testing.assert_allclose(pred.merf_calibrated_prediction, pred.merf_population_prediction + pred.b_hat)
zero = pred.n_calibration_samples == 0
np.testing.assert_allclose(pred.loc[zero, 'merf_population_prediction'], pred.loc[zero, 'merf_calibrated_prediction'])

for group in ['CU', 'AD', 'non-AD']:
    splits = json.loads((out / group / 'split_ids.json').read_text())
    assert not (set(splits['train']) & set(splits['validation']) or set(splits['train']) & set(splits['test']) or set(splits['validation']) & set(splits['test']))
    # This dataset had no ID/date/category/duplicate exclusions; compare exactly with the original recipe.
    if (audit['invalid_patient_id_rows'] == audit['invalid_visit_date_rows'] == audit['conflicting_category_patients'] == audit['same_day_duplicate_rows_removed'] == 0):
        g = raw[raw.bio_category == group]
        ids = g[g.patient_id.map(g.groupby('patient_id').size()) >= 2].patient_id.drop_duplicates().to_numpy()
        tr, tmp = train_test_split(ids, test_size=.30, random_state=42)
        va, te = train_test_split(tmp, test_size=.50, random_state=42)
        assert splits == {'train': tr.tolist(), 'validation': va.tolist(), 'test': te.tolist()}
    for r in pred[pred.Group == group].itertuples():
        patient = raw[raw.patient_id == r.patient_id]
        assert r.patient_id in splits['test']
        assert r.target_date == patient.visit_date.iloc[-1]
        assert r.true_mmse == patient.mmse.iloc[-1]
        assert r.last_mmse_prediction == patient.mmse.iloc[-2]
        assert r.n_calibration_samples == int(patient.mmse.iloc[1:-1].notna().sum())
    h = pd.read_csv(out / group / 'final_iterations.csv')
    assert len(h) == 20 and np.isfinite(h).all().all()
    info = json.loads((out / group / 'group_report.json').read_text())
    assert all(info['checks'].values())
print('PASS: executed notebook, original hash, exact original split recipe, raw test targets, aligned predictions, calibration counts, finite iteration history')
