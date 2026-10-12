"""Leakage-audited, native-NaN random-intercept MERF experiment.

No imputation, scaling, target clipping, or site-packages modifications.
MERF 1.0 passes X directly to its estimator: no validation adapter is needed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import platform
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from merf import MERF
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterGrid, train_test_split

SEED = 42
BLOOD = ['hdl', 'vldl', 'ldl', 'total_cholesterol', 'triglyceride', 'fasting_glucose', 'hba1c']
FEATURES = (
    ['mmse']
    + ['cdr_' + s for s in ['global', 'memory', 'orientation', 'judgment', 'community', 'home_hobbies', 'personal_care', 'sob']]
    + ['casi_' + s for s in ['mental_manipulation', 'attention', 'orientation', 'long_term_memory', 'short_term_memory', 'abstraction', 'drawing', 'verbal_fluency', 'language', 'total']]
    + ['days_since_casi', 'has_recent_casi'] + BLOOD
    + [b + '_gap_days_log' for b in BLOOD]
    + ['age_first_mmse', 'education', 'hypertension', 'diabetes', 'hyperlipidemia', 'apoe4_count', 'apoe_missing', 'gender', 'visit_gap_log']
)
GROUPS = ['CU', 'AD', 'non-AD']
GRID = {'n_estimators': [200], 'max_depth': [10], 'min_samples_leaf': [3, 5]}
MAX_ITERATIONS = 20  # fixed before viewing validation/test; no test-driven stopping
VAR_EPS = 1e-10


def save_json(path, value):
    def clean(x):
        if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [clean(v) for v in x]
        if isinstance(x, np.ndarray): return clean(x.tolist())
        if isinstance(x, np.generic): return clean(x.item())
        if isinstance(x, float) and not np.isfinite(x): return None
        if isinstance(x, (pd.Timestamp, Path)): return str(x)
        return x
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def versions():
    return {'python': platform.python_version(), 'executable': sys.executable,
            **{x: importlib.metadata.version(x) for x in ['numpy', 'pandas', 'scikit-learn', 'merf']}}


def prepare_data(csv, out):
    """Each row is a dated MMSE visit, with previously matched auxiliary data."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(csv)
    d['_source_row'] = np.arange(len(d))
    audit = {'input_rows': len(d), 'csv_sha256': hashlib.sha256(Path(csv).read_bytes()).hexdigest(),
             'row_semantics': 'one dated MMSE visit plus matched CDR/CASI/blood and demographic data',
             'mmse_scale': 'raw 0-30; preprocessing renames MMSE(numerical), no normalization',
             'existing_next_columns': [c for c in d if 'next' in c.lower()],
             'target_policy': 'rebuild from chronological mmse; never consume existing next columns',
             'numeric_issues': {}, 'future_auxiliary_masked': {}}
    events = []
    def event(rows, reason, action):
        for _, r in rows.iterrows():
            events.append({'source_row': r['_source_row'], 'patient_id': r['patient_id'],
                           'visit_date': r['visit_date'], 'reason': reason, 'action': action})
    # Original IDs are integer numeric identifiers. Preserve numeric sort order used by original notebook.
    ids = pd.to_numeric(d['patient_id'], errors='coerce')
    bad_id = ids.isna() | ~np.isfinite(ids) | (ids % 1 != 0) | (ids <= 0)
    d['visit_date'] = pd.to_datetime(d['visit_date'], errors='coerce', format='mixed').dt.normalize()
    audit['invalid_patient_id_rows'] = int(bad_id.sum())
    audit['invalid_visit_date_rows'] = int(d.visit_date.isna().sum())
    audit['date_columns'] = [c for c in d if c.endswith('_date')]
    event(d[bad_id], 'invalid_patient_id', 'exclude_row')
    event(d[d['visit_date'].isna()], 'invalid_visit_date', 'exclude_row')
    category_source = d.loc[~bad_id, ['patient_id', 'bio_category']].copy()
    category_source['patient_id'] = ids.loc[category_source.index].astype('int64')
    category_source['bio_category'] = category_source.bio_category.astype('string').str.strip().replace('', pd.NA)
    d = d.loc[~bad_id & d['visit_date'].notna()].copy()
    d['patient_id'] = ids.loc[d.index].astype('int64')
    # A conflicting category excludes the entire patient, rather than allocating visits across groups.
    d['bio_category'] = d['bio_category'].astype('string').str.strip().replace('', pd.NA)
    cats = category_source.groupby('patient_id')['bio_category'].agg(lambda s: sorted(s.dropna().unique().tolist()))
    conflict = cats[cats.map(len) > 1]
    audit['conflicting_category_patients'] = len(conflict)
    save_json(out / 'category_conflicts.json', conflict.to_dict())
    bad_patients = cats[cats.map(lambda x: len(x) != 1 or x[0] not in GROUPS)].index
    event(d[d.patient_id.isin(bad_patients)], 'conflicting_or_invalid_category', 'exclude_patient')
    d = d[~d.patient_id.isin(bad_patients)].copy()
    # Category is patient metadata, not a predictor. A missing label on a visit uses the single known label.
    category_assigned = int(d.bio_category.isna().sum())
    event(d[d.bio_category.isna()], 'missing_visit_category_single_patient_label', 'assign_patient_metadata_only')
    d['bio_category'] = d.patient_id.map(cats.map(lambda x: x[0] if len(x) == 1 else None))
    numeric = list(dict.fromkeys([c for c in FEATURES if c != 'visit_gap_log' and not c.endswith('_gap_days_log')]
                                + [b + '_gap_days' for b in BLOOD]))
    absent = set(numeric) - set(d.columns)
    if absent: raise ValueError(f'Missing required source columns: {sorted(absent)}')
    for c in numeric:
        v = pd.to_numeric(d[c], errors='coerce')
        invalid = d[c].notna() & v.isna()
        inf = np.isinf(v)
        audit['numeric_issues'][c] = {'non_numeric': int(invalid.sum()), 'infinity': int(inf.sum())}
        event(d[invalid | inf], 'invalid_numeric:' + c, 'set_NaN')
        d[c] = v.replace([np.inf, -np.inf], np.nan)
    invalid_mmse = d.mmse.notna() & ~d.mmse.between(0, 30)
    audit['out_of_range_MMSE_rows'] = int(invalid_mmse.sum())
    event(d[invalid_mmse], 'mmse_outside_0_30', 'set_NaN_keep_visit_for_history')
    d.loc[invalid_mmse, 'mmse'] = np.nan
    audit['mmse_observed_range'] = [d.mmse.min(), d.mmse.max()]
    # Source preprocessing used nearest blood matching, including future measurements.
    # Keep columns but mask any value unavailable AT that historical visit; never retrieve/fill an alternative.
    for b in BLOOD:
        datecol = b + '_blood_date'
        if datecol not in d: raise ValueError(f'Cannot audit availability without {datecol}')
        date = pd.to_datetime(d[datecol], errors='coerce', format='mixed').dt.normalize()
        bad = date.isna() | (date > d.visit_date)
        audit['future_auxiliary_masked'][b] = int((bad & d[b].notna()).sum())
        event(d[bad & d[b].notna()], 'unavailable_at_visit:' + b, 'mask_value_and_gap_NaN')
        d.loc[bad, [b, b + '_gap_days']] = np.nan
        # Original gap is blood_date - visit_date. Derive nonnegative elapsed age instead of clipping past gaps to 0.
        elapsed = (d.visit_date - date).dt.days.where(~bad & d[b].notna())
        d[b + '_gap_days_log'] = np.log1p(elapsed)
    date = pd.to_datetime(d['casi_date'], errors='coerce', format='mixed').dt.normalize()
    unavailable = date.isna() | (date > d.visit_date)
    audit['future_auxiliary_masked']['casi'] = int((unavailable & d.casi_total.notna()).sum())
    event(d[unavailable & d.casi_total.notna()], 'unavailable_at_visit:casi', 'mask_CASI_NaN')
    d.loc[unavailable, [c for c in FEATURES if c.startswith('casi_')] + ['days_since_casi', 'has_recent_casi']] = np.nan
    # No same-day aggregation: keep the first complete source ROW by stable CSV order, independent of MMSE.
    d = d.sort_values(['patient_id', 'visit_date', '_source_row'], kind='stable')
    dup = d.duplicated(['patient_id', 'visit_date'], keep='first')
    audit['same_day_duplicate_rows_removed'] = int(dup.sum())
    event(d[dup], 'same_day_duplicate', 'exclude_duplicate_keep_first_source_row_no_coalescing')
    d = d[~dup].reset_index(drop=True)
    gap = d.groupby('patient_id', sort=False).visit_date.diff().dt.days
    d['visit_gap_log'] = np.log1p(gap)  # first visit is NaN: no fill of an unknown previous gap
    audit.update({'clean_rows': len(d), 'clean_patients': d.patient_id.nunique(),
                  'missing_mmse_visits_kept': int(d.mmse.isna().sum()),
                  'missing_category_rows_assigned_single_patient_label': category_assigned,
                  'duplicate_policy': 'first source row on same calendar day; never combine fields',
                  'first_visit_gap_policy': 'NaN (no previous interval)',
                  'blood_gap_policy': 'log1p(visit_date - blood_date), only dates <= visit_date',
                  'feature_missing_rate': d[FEATURES].isna().mean().to_dict()})
    pd.DataFrame(events, columns=['source_row', 'patient_id', 'visit_date', 'reason', 'action']).to_csv(out / 'processing_events.csv', index=False)
    save_json(out / 'data_audit.json', audit)
    d[FEATURES].isna().mean().rename('missing_rate').to_csv(out / 'source_feature_missingness.csv')
    return d, audit


@dataclass
class Samples:
    X: pd.DataFrame
    y: np.ndarray
    meta: pd.DataFrame
    excluded: pd.DataFrame

    @property
    def Z(self): return np.ones((len(self.y), 1))

    @property
    def clusters(self): return self.meta.patient_id.reset_index(drop=True)


def create_tabular_history_features(data, feature_cols=FEATURES, final_only=False):
    rows, y, meta, excluded = [], [], [], []
    for pid, p in data.groupby('patient_id', sort=False):
        p = p.sort_values('visit_date', kind='stable').reset_index(drop=True)
        if len(p) < 2:
            excluded.append({'patient_id': pid, 'reason': 'fewer_than_two_visits', 'target_date': p.visit_date.iloc[-1]})
            continue
        indices = [len(p) - 1] if final_only else range(1, len(p))
        for j in indices:
            target = p.iloc[j]
            if pd.isna(target.mmse):
                excluded.append({'patient_id': pid, 'target_date': target.visit_date, 'reason': 'missing_target_MMSE_final' if final_only else 'missing_target_MMSE_transition'})
                continue
            h = p.iloc[:j]; v = h[feature_cols]
            assert h.visit_date.max() < target.visit_date
            row = {}
            stats = {'last': v.iloc[-1], 'mean': v.mean(), 'std': v.std(ddof=0), 'min': v.min(),
                     'max': v.max(), 'observed_count': v.count(), 'missing_rate': v.isna().mean()}
            for c in feature_cols:
                for stat, values in stats.items(): row[c + '_' + stat] = values[c]
            span = (h.visit_date.iloc[-1] - h.visit_date.iloc[0]).days
            row.update(history_n_visits=j, history_span_days=span)
            rows.append(row); y.append(float(target.mmse))
            meta.append({'patient_id': pid, 'history_end_date': h.visit_date.iloc[-1], 'target_date': target.visit_date,
                         'target_visit_index': j + 1, 'history_n_visits': j})
    cols = [c + '_' + s for c in feature_cols for s in ['last', 'mean', 'std', 'min', 'max', 'observed_count', 'missing_rate']] + ['history_n_visits', 'history_span_days']
    mc = ['patient_id', 'history_end_date', 'target_date', 'target_visit_index', 'history_n_visits']
    result = Samples(pd.DataFrame(rows, columns=cols, dtype=float), np.asarray(y), pd.DataFrame(meta, columns=mc),
                     pd.DataFrame(excluded, columns=['patient_id', 'target_date', 'reason']))
    assert not np.isinf(result.X.to_numpy()).any()
    assert np.isfinite(result.y).all() and np.isfinite(result.Z).all()
    assert not result.clusters.isna().any()
    return result


def split_patients(d):
    # Original notebook sorts patient_id/date, then drop_duplicates, then two sklearn splits.
    ids = d.patient_id.drop_duplicates().to_numpy()
    train, temp = train_test_split(ids, test_size=0.30, random_state=SEED)
    val, test = train_test_split(temp, test_size=0.50, random_state=SEED)
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
    return {'train': train.tolist(), 'validation': val.tolist(), 'test': test.tolist()}


def new_rf(params):
    return RandomForestRegressor(**params, random_state=SEED, n_jobs=-1)


def fit_merf(samples, params, iterations=MAX_ITERATIONS):
    model = MERF(fixed_effects_model=new_rf(params), max_iterations=iterations, gll_early_stop_threshold=None)
    model.fit(samples.X.to_numpy(), samples.Z, samples.clusters, samples.y)
    model_history(model)  # fail immediately if variances/iterations are not finite
    return model


def model_history(model):
    h = pd.DataFrame({'iteration': np.arange(1, len(model.gll_history) + 1), 'gll': model.gll_history,
                      'tau2': [float(np.asarray(d)[0, 0]) for d in model.D_hat_history[1:]],
                      'sigma2': model.sigma2_hat_history[1:]})
    assert np.isfinite(h.to_numpy()).all(), 'MERF returned non-finite iteration history'
    assert (h[['tau2', 'sigma2']] >= 0).all().all()
    assert np.isfinite(model.trained_b.to_numpy()).all()
    return h


def calibrated_predictions(model, targets, calibration):
    """Gaussian conditional mean using only strictly pre-target transition responses."""
    population = model.trained_fe_model.predict(targets.X.to_numpy())
    # Caller provides ONLY unseen patients in validation/test.
    assert not set(targets.clusters) & set(model.cluster_counts.index)
    native = model.predict(targets.X.to_numpy(), targets.Z, targets.clusters)
    np.testing.assert_allclose(native, population)
    tau = float(np.asarray(model.D_hat_history[-1])[0, 0]); sigma = float(model.sigma2_hat_history[-1])
    assert np.isfinite([tau, sigma]).all() and min(tau, sigma) >= 0
    residuals = calibration.y - model.trained_fe_model.predict(calibration.X.to_numpy()) if len(calibration.y) else np.array([])
    b, counts, flags = [], [], []
    for r in targets.meta.itertuples():
        mask = ((calibration.meta.patient_id == r.patient_id) & (calibration.meta.target_date < r.target_date)).to_numpy()
        m = int(mask.sum()); denominator = sigma + m * tau
        flag = 'none'
        if m == 0: intercept = 0.0
        elif tau <= VAR_EPS: intercept = 0.0; flag = 'tau2_near_zero_population_fallback'
        elif denominator <= VAR_EPS: intercept = 0.0; flag = 'denominator_near_zero_population_fallback'
        else:
            intercept = tau / denominator * residuals[mask].sum()
            if sigma <= VAR_EPS: flag = 'sigma2_near_zero'
        b.append(intercept); counts.append(m); flags.append(flag)
    b = np.asarray(b)
    assert np.isfinite(population).all() and np.isfinite(b).all()
    np.testing.assert_allclose((population + b)[np.array(counts) == 0], population[np.array(counts) == 0])
    return population, population + b, b, counts, flags


def metrics(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    valid = np.isfinite(y) & np.isfinite(prediction)
    y, prediction = y[valid], prediction[valid]
    nz = y != 0
    return {'n': len(y), 'MAE': mean_absolute_error(y, prediction) if len(y) else np.nan,
            'RMSE': np.sqrt(mean_squared_error(y, prediction)) if len(y) else np.nan,
            'R2': r2_score(y, prediction) if len(y) > 1 else np.nan,
            'MAPE_percent': np.mean(np.abs((y[nz] - prediction[nz]) / y[nz])) * 100 if nz.any() else np.nan,
            'MAPE_zero_excluded': int((~nz).sum())}


PRED_COLS = ['last_mmse_prediction', 'rf_prediction', 'merf_population_prediction', 'merf_calibrated_prediction']


def evaluation_tables(pred):
    rows = []
    for group, p in pred.groupby('Group', sort=False):
        for scope, subset in [('all', p), ('common_baseline_available', p[p.last_mmse_prediction.notna()])]:
            strata = [('all', subset), ('0', subset[subset.n_calibration_samples == 0]),
                      ('1', subset[subset.n_calibration_samples == 1]), ('2+', subset[subset.n_calibration_samples >= 2])]
            for stratum, s in strata:
                for col in PRED_COLS:
                    rows.append({'Group': group, 'scope': scope, 'calibration_stratum': stratum,
                                 'model': col, 'eligible_targets': len(s), **metrics(s.true_mmse, s[col])})
    return pd.DataFrame(rows)


def check_final_target_invariance(d, targets, calibration, model, feature_cols=FEATURES):
    changed = d.copy()
    for r in targets.meta.itertuples():
        mask = (changed.patient_id == r.patient_id) & (changed.visit_date == r.target_date)
        changed.loc[mask, 'mmse'] = (float(changed.loc[mask, 'mmse'].iloc[0]) + 7) % 31
    altered = create_tabular_history_features(changed, feature_cols, final_only=True)
    altered_cal = calibration_samples(changed, altered, feature_cols)
    pd.testing.assert_frame_equal(targets.X, altered.X)
    pd.testing.assert_frame_equal(calibration.X, altered_cal.X)
    np.testing.assert_allclose(calibration.y, altered_cal.y)
    before = calibrated_predictions(model, targets, calibration)
    after = calibrated_predictions(model, altered, altered_cal)
    np.testing.assert_allclose(before[2], after[2])
    return True


def calibration_samples(d, targets, feature_cols=FEATURES):
    # Explicit separate construction: remove final target rows BEFORE building transitions.
    cutoffs = targets.meta.set_index('patient_id').target_date
    cutoff = d.patient_id.map(cutoffs)
    history = d[cutoff.notna() & (d.visit_date < cutoff)].copy()
    return create_tabular_history_features(history, feature_cols, final_only=False)


def synthetic_checks(out):
    # Exercise real cleaning policy with deliberately malformed source records.
    base = {c: 1.0 for c in FEATURES if not c.endswith('_log')}
    base.update(patient_id=100, visit_date='2020-01-01', mmse=25.0, bio_category='CU', casi_date='2019-12-01')
    for blood in BLOOD:
        base[blood + '_gap_days'] = -31.0
        base[blood + '_blood_date'] = '2019-12-01'
    fixture = [base.copy()]
    fixture.append({**base, 'mmse': 30.0})  # duplicate: first row must survive unchanged
    fixture.append({**base, 'visit_date': '2020-06-01', 'mmse': 31.0, 'education': 'bad', 'hdl': np.inf,
                    'ldl_blood_date': '2020-07-01'})
    fixture.extend([{**base, 'patient_id': 102, 'bio_category': c} for c in ['CU', 'AD']])
    fixture.extend([{**base, 'patient_id': 'bad'}, {**base, 'patient_id': 103, 'visit_date': 'bad'}])
    with tempfile.TemporaryDirectory(dir=out, prefix='synthetic_cleaning_') as temp:
        fixture_csv = Path(temp) / 'fixture.csv'
        pd.DataFrame(fixture).to_csv(fixture_csv, index=False)
        cleaned, cleaning_audit = prepare_data(fixture_csv, temp)
        assert len(cleaned) == 2 and cleaned.patient_id.eq(100).all()
        assert cleaned.mmse.iloc[0] == 25 and pd.isna(cleaned.mmse.iloc[1])
        assert pd.isna(cleaned.education.iloc[1]) and pd.isna(cleaned.hdl.iloc[1])
        assert pd.isna(cleaned.ldl.iloc[1]) and pd.isna(cleaned.visit_gap_log.iloc[0])
        assert cleaning_audit['conflicting_category_patients'] == 1
        assert cleaning_audit['same_day_duplicate_rows_removed'] == 1
        assert cleaning_audit['invalid_patient_id_rows'] == 1 and cleaning_audit['invalid_visit_date_rows'] == 1
    rng = np.random.default_rng(SEED)
    rows = []
    for pid in range(16):
        intercept = rng.normal(0, 1.5)
        for j in range(2 + pid % 4):
            rows.append({'patient_id': pid, 'visit_date': pd.Timestamp('2020-01-01') + pd.Timedelta(days=180*j),
                         'mmse': 24 + intercept - j + rng.normal(0, .2), 'partial': np.nan if j % 2 else pid / 10,
                         'all_missing': np.nan})
    d = pd.DataFrame(rows); cols = ['mmse', 'partial', 'all_missing']
    train = create_tabular_history_features(d[d.patient_id < 12], cols)
    test_d = d[d.patient_id >= 12].copy()
    test = create_tabular_history_features(test_d, cols, True)
    cal = calibration_samples(test_d, test, cols)
    assert train.X.isna().any().any() and train.X.all_missing_last.isna().all()
    assert train.X.all_missing_mean.isna().all()
    assert train.X.all_missing_std.isna().all()
    assert (train.X.mmse_std[train.X.mmse_observed_count == 1] == 0).all()
    assert (train.X.partial_last.isna() & train.X.partial_mean.notna()).any()
    # Direct raw-X fit/predict, including a literal entirely missing column.
    raw = train.X.to_numpy(); params = {'n_estimators': 12, 'max_depth': 4, 'min_samples_leaf': 2}
    rf = new_rf(params).fit(raw, train.y)
    assert np.isfinite(rf.predict(test.X.to_numpy())).all()
    model = fit_merf(train, params, iterations=3)
    result = calibrated_predictions(model, test, cal)
    assert result[3] == [0, 1, 2, 3]  # first visit never supplies a residual
    tau = float(model.D_hat_history[-1][0, 0]); sigma = float(model.sigma2_hat_history[-1])
    residual = cal.y - model.trained_fe_model.predict(cal.X.to_numpy())
    for i, r in enumerate(test.meta.itertuples()):
        mask = ((cal.meta.patient_id == r.patient_id) & (cal.meta.target_date < r.target_date)).to_numpy()
        expected = tau / (sigma + mask.sum() * tau) * residual[mask].sum()
        np.testing.assert_allclose(result[2][i], expected)
    saved_tau, saved_sigma = model.D_hat_history[-1], model.sigma2_hat_history[-1]
    try:
        model.D_hat_history[-1] = np.array([[0.0]])
        model.sigma2_hat_history[-1] = 0.0
        near_zero = calibrated_predictions(model, test, cal)
        np.testing.assert_allclose(near_zero[0], near_zero[1])
        assert all(f != 'none' for f in near_zero[4][1:])
    finally:
        model.D_hat_history[-1], model.sigma2_hat_history[-1] = saved_tau, saved_sigma
    check_final_target_invariance(test_d, test, cal, model, cols)
    # Missing final target is excluded, never replaced by an earlier target.
    missing = test_d.copy(); idx = missing.groupby('patient_id').tail(1).index[0]
    missing.loc[idx, 'mmse'] = np.nan
    reduced = create_tabular_history_features(missing, cols, True)
    assert len(reduced.y) == len(test.y) - 1
    # Native known-patient predict also exercises the package wrapper.
    assert np.isfinite(model.predict(raw, train.Z, train.clusters)).all()
    result = {'passed': True, 'RF_native_NaN_fit_predict': True, 'MERF_native_NaN_fit_predict': True,
              'all_missing_column_preserved': True, 'final_target_invariance': True,
              'missing_final_target_no_fallback': True, 'two_visit_calibration_count': 0,
              'conditional_mean_formula': True, 'near_zero_variance_handled': True,
              'source_cleaning_invalid_ids_dates_duplicates_categories_values': True,
              'adapter': 'not required: merf 1.0 passes NaN X directly to sklearn RF', 'versions': versions()}
    save_json(Path(out) / 'synthetic_checks.json', result)
    return result


def run_group(d, group, out, smoke=False):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    g = d[d.bio_category == group].copy()
    # Match original eligibility (>=2 visits), but retain visits with missing MMSE.
    eligible = g.groupby('patient_id').size(); excluded_ids = eligible[eligible < 2].index.tolist()
    save_json(out / 'single_visit_excluded_ids.json', excluded_ids)
    g = g[g.patient_id.isin(eligible[eligible >= 2].index)]
    splits = split_patients(g); save_json(out / 'split_ids.json', splits)
    frames = {k: g[g.patient_id.isin(v)] for k, v in splits.items()}
    train = create_tabular_history_features(frames['train'])
    val = create_tabular_history_features(frames['validation'], final_only=True)
    test = create_tabular_history_features(frames['test'], final_only=True)
    val_cal = calibration_samples(frames['validation'], val)
    test_cal = calibration_samples(frames['test'], test)
    for name, samples in [('train', train), ('validation', val), ('test', test), ('validation_calibration', val_cal), ('test_calibration', test_cal)]:
        samples.meta.to_csv(out / (name + '_sample_index.csv'), index=False)
        samples.excluded.to_csv(out / (name + '_excluded.csv'), index=False)
        samples.X.isna().mean().rename('missing_rate').to_csv(out / (name + '_X_missingness.csv'))
    if min(len(train.y), len(val.y), len(test.y)) == 0: raise ValueError(f'{group}: empty eligible split')
    assert train.X.columns.equals(val.X.columns) and train.X.columns.equals(test.X.columns)
    if smoke:
        small = Samples(train.X.iloc[:40], train.y[:40], train.meta.iloc[:40].reset_index(drop=True), train.excluded)
        small_params = {'n_estimators': 12, 'max_depth': 4, 'min_samples_leaf': 3}
        rf = new_rf(small_params).fit(small.X.to_numpy(), small.y)
        rf_pred = rf.predict(val.X.to_numpy())
        assert np.isfinite(rf_pred).all()
        m = fit_merf(small, small_params, 2)
        pred = calibrated_predictions(m, val, val_cal)
        save_json(out / 'smoke.json', {'passed': True, 'training_rows': len(small.y), 'validation_rows': len(val.y),
                                     'RF_validation_RMSE': metrics(val.y, rf_pred)['RMSE'],
                                     'validation_RMSE': metrics(val.y, pred[1])['RMSE']})
        return None
    tuning = []
    for i, params in enumerate(ParameterGrid(GRID)):
        print(f'{group}: candidate {i + 1} RF + MERF', flush=True)
        rf = new_rf(params).fit(train.X.to_numpy(), train.y)
        tuning.append({'model': 'RF', 'candidate': i, **params, **metrics(val.y, rf.predict(val.X.to_numpy()))})
        m = fit_merf(train, params)
        pop, calibrated, *_ = calibrated_predictions(m, val, val_cal)
        h = model_history(m); h.to_csv(out / f'candidate_{i}_iterations.csv', index=False)
        tuning.append({'model': 'MERF_history_calibrated', 'candidate': i, **params,
                       **metrics(val.y, calibrated), 'population_RMSE': metrics(val.y, pop)['RMSE']})
        pd.DataFrame(tuning).to_csv(out / 'tuning.csv', index=False)
    t = pd.DataFrame(tuning)
    candidates = list(ParameterGrid(GRID))
    best = {name: candidates[int(t[t.model == name].sort_values(['RMSE', 'candidate']).iloc[0].candidate)]
            for name in ['RF', 'MERF_history_calibrated']}
    save_json(out / 'selected_parameters.json', {'params': best, 'MERF_max_iterations': MAX_ITERATIONS,
                                               'selection': 'validation RMSE, MERF history-calibrated; test untouched'})
    final_d = pd.concat([frames['train'], frames['validation']]).sort_values(['patient_id', 'visit_date'])
    final = create_tabular_history_features(final_d)
    final.meta.to_csv(out / 'final_training_sample_index.csv', index=False)
    final.excluded.to_csv(out / 'final_training_excluded.csv', index=False)
    rf = new_rf(best['RF']).fit(final.X.to_numpy(), final.y)
    m = fit_merf(final, best['MERF_history_calibrated'])
    h = model_history(m); h.to_csv(out / 'final_iterations.csv', index=False)
    pop, calibrated, b, counts, flags = calibrated_predictions(m, test, test_cal)
    assert check_final_target_invariance(frames['test'], test, test_cal, m)
    pred = test.meta.copy()
    pred.insert(0, 'Group', group)
    pred['true_mmse'] = test.y
    pred['last_mmse_prediction'] = test.X.mmse_last.to_numpy()  # latest visit, never forward-filled
    pred['rf_prediction'] = rf.predict(test.X.to_numpy())
    pred['merf_population_prediction'] = pop
    pred['merf_calibrated_prediction'] = calibrated
    pred['b_hat'] = b; pred['n_calibration_samples'] = counts; pred['variance_degeneracy'] = flags
    assert np.isfinite(pred[PRED_COLS[1:] + ['true_mmse', 'b_hat']].to_numpy()).all()
    assert len(pred) == len(test.y) and not pred.patient_id.duplicated().any()
    pred.to_csv(out / 'test_predictions.csv', index=False)
    evaluation_tables(pred).to_csv(out / 'metrics.csv', index=False)
    relative = float(abs(h.gll.iloc[-1] - h.gll.iloc[-2]) / max(abs(h.gll.iloc[-2]), VAR_EPS))
    info = {'group': group, 'split_patients': {k: len(v) for k, v in splits.items()},
            'train_transitions': len(train.y), 'final_train_transitions': len(final.y),
            'validation_targets': len(val.y), 'test_targets': len(test.y),
            'test_calibration_transitions': len(test_cal.y), 'test_excluded_targets': len(test.excluded),
            'baseline_missing': int(pred.last_mmse_prediction.isna().sum()),
            'baseline_coverage': float(pred.last_mmse_prediction.notna().mean()),
            'training_X_missing_rate': float(final.X.isna().to_numpy().mean()),
            'test_X_missing_rate': float(test.X.isna().to_numpy().mean()),
            'tau2': h.tau2.iloc[-1], 'sigma2': h.sigma2.iloc[-1], 'iterations': len(h),
            'final_GLL_relative_change': relative,
            'convergence_diagnostic': 'relative change < 0.001' if relative < .001 else 'iteration budget reached; convergence not established',
            'GLL_increase_steps': int((h.gll.diff() > 0).sum()),
            'degenerate_variance_predictions': int((pred.variance_degeneracy != 'none').sum()),
            'checks': {'patient_split_disjoint': True, 'strict_history_dates': True, 'aligned_prediction_rows': True,
                       'final_target_X_and_b_invariance': True, 'finite_model_and_prediction': True,
                       'unseen_native_equals_fixed_effect': True, 'zero_calibration_equals_population': True,
                       'X_NaN_retained': bool(final.X.isna().any().any()), 'y_Z_finite': True}}
    save_json(out / 'group_report.json', info)
    print(group, 'completed:', info['test_targets'], 'test targets', flush=True)
    return pred, info


def write_report(out, audit, reports, table):
    text = ['# MERF 實驗報告', '', '真實 CSV 實驗已完成。MMSE 與所有預測皆為原始分數，未四捨五入或裁切。',
            f"原始 {audit['input_rows']} 列；清理後 {audit['clean_rows']} 列、{audit['clean_patients']} 人。",
            '每列是 MMSE visit；CSV 沒有直接用作標籤的 next target。上游 MMSE 由原始 numerical 欄改名，範圍 0–30。',
            '同日保留原始第一列，不合併欄位；跨 bio_category 病人整人排除。缺失 MMSE visit 保留作歷史，缺失 target 排除。',
            '上游 nearest blood 配對可能取未來檢驗：本流程將晚於該 visit 或日期不明的血液值與 gap 遮罩為 NaN。',
            '這不會重建真正的過去檢驗；因此缺失率可能提高。血液 gap 改為已發生檢驗的 elapsed days log，第一筆 visit gap 保留 NaN。',
            '', 'MERF 1.0 + sklearn RF 原生 NaN 完整 fit/predict 通過，包含整欄 NaN；不需 adapter，也未修改 site-packages。',
            '條件 random intercept 校正依 Gaussian 假設，相關歷史殘差並非額外訓練資料；tau²、sigma² 僅由 training fit 取得。',
            '固定 20 iterations，兩組小型 grid，以 validation calibrated RMSE 選 MERF；選後 train+validation 全 transitions 重訓。',
            'GLL 是套件的目標診斷量（較低較好），RF EM 不保證單調；僅依最後相對變化描述，不保證收斂。', '']
    for r in reports:
        text += [f"## {r['group']}", '',
                 f"病人切分 {r['split_patients']}；train transitions={r['train_transitions']}，最終重訓={r['final_train_transitions']}。",
                 f"validation targets={r['validation_targets']}，test targets={r['test_targets']}，test 排除={r['test_excluded_targets']}。",
                 f"X 缺失率 train={r['training_X_missing_rate']:.2%}、test={r['test_X_missing_rate']:.2%}；baseline 覆蓋率={r['baseline_coverage']:.2%}。",
                 f"tau²={r['tau2']:.6g}，sigma²={r['sigma2']:.6g}，iterations={r['iterations']}，最後 GLL 相對變化={r['final_GLL_relative_change']:.6g}；{r['convergence_diagnostic']}。", '']
    text += ['## 模型比較（各組所有 test targets；baseline 只評估可用者）', '',
             table[(table.scope == 'all') & (table.calibration_stratum == 'all')].to_string(index=False), '',
             '公平比較請見 metrics.csv 的 common_baseline_available，另有校正樣本數 0／1／2+ 分層。',
             '資料處理逐列原因見 processing_events.csv；split／調參／迭代／逐病人輸出均在各組資料夾。',
             '人工檢查與真實資料最終 target 擾動不影響 X 或 b_hat 的檢查皆通過。']
    (Path(out) / 'report_zh.md').write_text('\n'.join(text), encoding='utf-8')


def run_experiment(root, out=None):
    root = Path(root); out = Path(out) if out else root / 'results/merf_nan_CU_AD_nonAD'
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / 'environment.json', versions())
    synthetic_checks(out)
    csv = root / 'data/processed/model_df_next_mmse3.csv'
    if not csv.exists():
        save_json(out / 'execution_status.json', {'synthetic': 'passed', 'real_data': 'not_run_missing_csv', 'path': str(csv)})
        raise FileNotFoundError(f'Artificial checks passed; real experiment NOT run: {csv}')
    d, audit = prepare_data(csv, out)
    for group in GROUPS: run_group(d, group, out / 'smoke' / group, smoke=True)
    results = [run_group(d, group, out / group) for group in GROUPS]
    predictions = pd.concat([r[0] for r in results], ignore_index=True)
    table = evaluation_tables(predictions)
    predictions.to_csv(out / 'test_predictions.csv', index=False)
    table.to_csv(out / 'metrics.csv', index=False)
    write_report(out, audit, [r[1] for r in results], table)
    save_json(out / 'execution_status.json', {'synthetic': 'passed', 'smoke': 'passed', 'real_data': 'completed'})
    return predictions, table


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--synthetic-only', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    if args.synthetic_only:
        out = args.root / 'results/merf_nan_CU_AD_nonAD'; out.mkdir(parents=True, exist_ok=True)
        print(synthetic_checks(out))
    else: run_experiment(args.root)
