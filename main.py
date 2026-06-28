# main.py
# ─────────────────────────────────────────────────────────────
# I²DARS — TEMPORARY VERSION
# Training Detection Ensemble ONLY (without RL agent)
# RL agent training will be added later
# ─────────────────────────────────────────────────────────────

import argparse
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from config import *
from data.preprocess import preprocess
from models.ensemble.train_ensemble import train_full_ensemble
from models.ensemble.predict import load_models, predict_batch
from models.explainability.shap_engine import (compute_shap_tree,
                                                get_top_features,
                                                plot_shap_bar)
from models.explainability.mitre_mapper import map_to_mitre, generate_full_report
from models.rl_agent.safety_shield import SafetyShield
from evaluation.metrics import (compute_metrics, print_table,
                                plot_confusion_matrix,
                                plot_comparison)


# ── Helper Functions ────────────────────────────────────────

def _load_saved():
    """Load all saved models and artifacts."""
    import joblib
    
    print('[*] Loading saved models and artifacts...')
    
    # Check if models exist
    required_files = [
        RF_MODEL_PATH, XGB_MODEL_PATH, DNN_MODEL_PATH,
        CATBOOST_MODEL_PATH, SCALER_PATH, ENCODER_PATH,
        FEATURE_COLS_PATH
    ]
    
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print(f'[ERROR] Missing required files:')
        for f in missing:
            print(f'  - {f}')
        print(f'\nPlease run: python main.py --mode train')
        raise FileNotFoundError('Required model files not found')
    
    rf, xgb_m, dnn, catboost = load_models()
    scaler = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODER_PATH)
    
    with open(FEATURE_COLS_PATH) as fh:
        feat_cols = json.load(fh)
    
    X_test = np.load(os.path.join(MODEL_DIR, 'X_test.npy'))
    y_test = np.load(os.path.join(MODEL_DIR, 'y_test.npy'))
    
    print('[✓] All models loaded successfully')
    return rf, xgb_m, dnn, catboost, scaler, encoders, feat_cols, X_test, y_test


def print_banner(mode: str):
    """Print system banner."""
    banner = (
        '\n' + '█' * 70 + '\n'
        '  I²DARS — TEMPORARY VERSION\n'
        '  Detection Ensemble Training (RL Agent Training Deferred)\n'
        f'  Mode : {mode.upper()}\n'
        '█' * 70 + '\n'
    )
    print(banner)


# ── Phase 1: Training ───────────────────────────────────────

def phase_train():
    """
    PHASE 1: Data preprocessing and ensemble model training.
    
    ⚠ TEMPORARY VERSION: Skips RL agent training
    
    Includes:
    - Data loading and preprocessing (CSE-CIC-IDS2018)
    - Stacked ensemble training (RF + XGB + DNN + CatBoost)
    
    RL agent training will be added after ensemble is verified.
    """
    print('\n' + '=' * 70)
    print('  PHASE 1 — DATA PREPROCESSING & ENSEMBLE TRAINING')
    print('=' * 70)
    print('  [TEMPORARY] RL agent training skipped\n')
    
    # ── Step 1: Data Preprocessing ──────────────────────────
    print('[STEP 1] Data Preprocessing\n')
    print('[*] Loading CSE-CIC-IDS2018 dataset...')
    
    try:
        X_tr, X_va, X_te, y_tr, y_va, y_te, _, _, _ = preprocess(
            data_dir=DATA_DIR,
            file_list=DATA_FILES,
            sample_size=1.0  # Change to 0.1 for faster testing
        )
    except Exception as e:
        print(f'\n[ERROR] Failed to preprocess data:')
        print(f'  {e}')
        print(f'\nTroubleshooting:')
        print(f'  1. Check that CSV files are in: {DATA_DIR}')
        print(f'  2. Verify filenames match DATA_FILES in config.py')
        print(f'  3. Ensure CSV files are not corrupted')
        raise
    
    # ── Step 2: Ensemble Training ───────────────────────────
    print('\n[STEP 2] Training Stacked Ensemble\n')
    try:
        train_full_ensemble(X_tr, y_tr, X_va, y_va)
    except Exception as e:
        print(f'\n[ERROR] Ensemble training failed:')
        print(f'  {e}')
        raise
    
    print('\n' + '=' * 70)
    print('[✓] PHASE 1 COMPLETE - Ensemble training finished')
    print('=' * 70)
    print('\n[NEXT STEPS]')
    print('  1. Run evaluation: python main.py --mode evaluate')
    print('  2. Run demo: python main.py --mode demo')
    print('  3. Later, train RL agent: python main.py --mode train_rl\n')


# ── Phase 2: Evaluation ─────────────────────────────────────

def phase_evaluate():
    """
    PHASE 2: Ensemble model evaluation and analysis.
    
    Generates:
    - Confusion matrix
    - Metrics comparison charts
    - SHAP importance plots
    - Ablation study results
    - Per-class performance breakdown
    """
    print('\n' + '=' * 70)
    print('  PHASE 2 — ENSEMBLE EVALUATION & ANALYSIS')
    print('=' * 70)
    
    os.makedirs('outputs', exist_ok=True)
    
    print('\n[*] Loading trained models...')
    rf, xgb_m, dnn, catboost, _, _, feat_cols, X_te, y_te = _load_saved()
    
    # ── Step 1: Detection Performance ───────────────────────
    print('\n[STEP 1] Evaluating Detection Performance\n')
    
    results = predict_batch(X_te, rf, xgb_m, dnn, catboost)
    inv = {v: k for k, v in ATTACK_NAMES.items()}
    fp_i = np.array([inv[r['final_prediction']] for r in results])
    
    # Compute metrics for each model
    all_m = [
        compute_metrics(y_te, rf.predict(X_te),
                       'Random Forest'),
        compute_metrics(y_te, xgb_m.predict(X_te),
                       'XGBoost'),
        compute_metrics(y_te,
                       np.argmax(dnn.predict(X_te, verbose=0), axis=1),
                       'DNN'),
        compute_metrics(y_te, fp_i,
                       'I²DARS Ensemble'),
    ]
    
    print('\n[RESULTS] Model Performance Comparison:\n')
    print_table(all_m)
    
    # ── Step 2: Confusion Matrix ────────────────────────────
    print('\n[STEP 2] Generating Confusion Matrix\n')
    plot_confusion_matrix(y_te, fp_i, 'I²DARS Ensemble')
    
    # ── Step 3: Performance Comparison Chart ────────────────
    print('\n[STEP 3] Generating Performance Comparison Chart\n')
    plot_comparison(all_m)
    
    # ── Step 4: SHAP Analysis ──────────────────────────────
    print('\n[STEP 4] Computing SHAP Feature Importance\n')
    print('[*] Computing SHAP values on first 200 test samples...')
    print('    (This may take 10-15 minutes on first run)\n')
    
    try:
        shap_v, _ = compute_shap_tree(rf, X_te[:200], feat_cols)
        plot_shap_bar(shap_v, feat_cols, 0,
                     save_path='outputs/shap_summary.png',
                     title='SHAP Feature Importance (Random Forest)')
        print('[✓] SHAP analysis complete')
    except Exception as e:
        print(f'[!] SHAP computation warning: {e}')
        print(f'    Continuing without SHAP visualizations...')
    
    # ── Step 5: Ablation Study ─────────────────────────────
    print('\n[STEP 5] Running Ablation Study\n')
    try:
        print('[*] Testing ensemble without each base learner...')
        
        # Without DNN
        rf_p = rf.predict_proba(X_te)
        xgb_p = xgb_m.predict_proba(X_te)
        zero = np.zeros_like(rf_p)
        meta_nd = np.hstack([rf_p, xgb_p, zero])
        nd_preds = np.argmax(catboost.predict_proba(meta_nd), axis=1)
        ablation_nd = compute_metrics(y_te, nd_preds, 'Without DNN')
        
        # Without XGBoost
        dnn_p = dnn.predict(X_te, verbose=0)
        meta_nx = np.hstack([rf_p, zero, dnn_p])
        nx_preds = np.argmax(catboost.predict_proba(meta_nx), axis=1)
        ablation_nx = compute_metrics(y_te, nx_preds, 'Without XGBoost')
        
        # Without Random Forest
        zero_rf = np.zeros_like(xgb_p)
        meta_nrf = np.hstack([zero_rf, xgb_p, dnn_p])
        nrf_preds = np.argmax(catboost.predict_proba(meta_nrf), axis=1)
        ablation_nrf = compute_metrics(y_te, nrf_preds, 'Without Random Forest')
        
        ablation_results = [
            all_m[3],  # Full ensemble
            ablation_nd,
            ablation_nx,
            ablation_nrf,
        ]
        
        print('\n[RESULTS] Ablation Study:\n')
        print_table(ablation_results)
        
    except Exception as e:
        print(f'[!] Ablation study error: {e}')
    
    print('\n' + '=' * 70)
    print('[✓] PHASE 2 COMPLETE - Evaluation finished')
    print('=' * 70)
    print(f'[*] All plots saved to: outputs/')


# ── Phase 3: End-to-End Demo ────────────────────────────────

def phase_demo(n: int = 5):
    """
    PHASE 3: End-to-end detection pipeline demonstration.
    
    ⚠ TEMPORARY VERSION: Shows detection and SHAP only (no RL)
    
    Shows:
    - Detection on sample threats
    - SHAP explanations
    - MITRE ATT&CK mapping
    - Per-sample incident reports
    """
    print('\n' + '=' * 70)
    print('  PHASE 3 — DETECTION & EXPLANATION DEMO')
    print('=' * 70)
    print('  [TEMPORARY] RL agent demo skipped\n')
    
    os.makedirs('outputs', exist_ok=True)
    
    print('[*] Loading models...')
    rf, xgb_m, dnn, catboost, _, _, feat_cols, X_te, y_te = _load_saved()
    
    inv = {v: k for k, v in ATTACK_NAMES.items()}
    
    # Select demo samples (one per attack class)
    print(f'\n[*] Selecting {min(n, len(np.unique(y_te)))} representative threat samples...\n')
    
    demo_idx = []
    for lbl in sorted(np.unique(y_te)):
        idx = np.where(y_te == lbl)[0]
        if len(idx) > 0:
            demo_idx.append(int(idx[0]))
        if len(demo_idx) >= n:
            break
    
    all_reports = []
    
    for rank, row_i in enumerate(demo_idx, 1):
        x = X_te[row_i : row_i + 1]
        true_label = ATTACK_NAMES[y_te[row_i]]
        
        print('─' * 70)
        print(f'  SAMPLE {rank}/{len(demo_idx)} — True Label: {true_label}')
        print('─' * 70)
        
        # ── Step 1: Detection ───────────────────────────────
        print('\n  [DETECTION]')
        det = predict_batch(x, rf, xgb_m, dnn, catboost)[0]
        print(f'  Predicted: {det["final_prediction"]} ({det["final_confidence"]}%)')
        print(f'  Threat detected: {"YES" if det["is_threat"] else "NO"}')
        
        print(f'\n  Base model predictions:')
        for mdl, data in det['base_models'].items():
            print(f'    {mdl:<20}: {data["prediction"]:<25} ({data["confidence"]}%)')
        
        pred_cls = inv[det['final_prediction']]
        
        # ── Step 2: SHAP + MITRE ────────────────────────────
        print(f'\n  [EXPLAINABILITY]')
        
        try:
            shap_v, _ = compute_shap_tree(rf, x, feat_cols)
            top_feats = get_top_features(shap_v, feat_cols, pred_cls, top_n=5)
            
            print(f'  Top SHAP Features:')
            for f in top_feats:
                arrow = '↑ anomalous' if f['direction'] == 'positive' else '↓ normal'
                print(f'    • {f["feature"]:<35} SHAP={f["shap_value"]:+.4f} ({arrow})')
            
            mitre = map_to_mitre(det['final_prediction'], top_feats)
            
            if mitre:
                print(f'\n  [MITRE ATT&CK]')
                print(f'  Tactic    : {mitre["tactic"]}')
                print(f'  Technique : {mitre["technique_id"]} — {mitre["technique_name"]}')
                if mitre.get('matched_features'):
                    print(f'  Matched indicators: {", ".join(mitre["matched_features"])}')
            
            # Save SHAP plot for this sample
            plot_shap_bar(shap_v, feat_cols, pred_cls,
                         save_path=f'outputs/shap_sample_{rank}.png',
                         title=f'SHAP Attribution — {det["final_prediction"]}')
            
        except Exception as e:
            print(f'  [!] SHAP computation skipped: {e}')
            mitre = None
            top_feats = None
        
        # ── Step 3: Full Report ─────────────────────────────
        print(f'\n  [INCIDENT REPORT]')
        
        rpt = generate_full_report(det, mitre, None, row_i)  # No RL data
        all_reports.append(rpt)
        
        rpt_path = f'outputs/report_sample_{rank}.txt'
        with open(rpt_path, 'w') as fh:
            fh.write(rpt)
        
        print(f'  Report saved → {rpt_path}')
    
    print(f'\n' + '=' * 70)
    print(f'[✓] PHASE 3 COMPLETE - Demo finished')
    print('=' * 70)
    print(f'[*] Generated {len(all_reports)} incident reports')
    print(f'[*] All outputs saved to: outputs/')
    print(f'\n[NOTE] This demo shows detection & explanation only.')
    print(f'       RL autonomous response will be added after RL training.')


# ── Main Entry Point ────────────────────────────────────────

def main():
    """
    Main entry point for I²DARS pipeline (TEMPORARY - DETECTION ONLY).
    
    ⚠ TEMPORARY VERSION: Trains detection ensemble only
       RL agent training deferred for later
    
    Usage:
        python main.py --mode full              # Train + evaluate + demo
        python main.py --mode train             # Train ensemble only
        python main.py --mode evaluate          # Evaluate ensemble only
        python main.py --mode demo              # Demo detection only
    """
    
    parser = argparse.ArgumentParser(
        description='I²DARS — Detection Ensemble Training (Temporary)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
⚠ TEMPORARY VERSION: Trains detection ensemble only

Examples:
  # Train ensemble
  python main.py --mode train
  
  # Train + evaluate + demo (full pipeline without RL)
  python main.py --mode full
  
  # Evaluate trained ensemble
  python main.py --mode evaluate
  
  # Run demo (detection + explanation only)
  python main.py --mode demo --demo_samples 10

After ensemble training is complete, RL agent can be trained separately:
  python main.py --mode train_rl --rl_episodes 5000
        '''
    )
    
    parser.add_argument(
        '--mode',
        default='full',
        choices=['train', 'evaluate', 'demo', 'full'],
        help='Which phase(s) to run (RL training deferred)'
    )
    
    parser.add_argument(
        '--demo_samples',
        type=int,
        default=5,
        help='Number of samples for demo phase'
    )
    
    args = parser.parse_args()
    
    # ── Print Banner ────────────────────────────────────────
    print_banner(args.mode)
    
    try:
        # ── Execute Phases ─────────────────────────────────
        if args.mode in ('train', 'full'):
            phase_train()
        
        if args.mode in ('evaluate', 'full'):
            phase_evaluate()
        
        if args.mode in ('demo', 'full'):
            phase_demo(args.demo_samples)
        
        # ── Final Summary ───────────────────────────────────
        print('\n' + '█' * 70)
        print('  PIPELINE EXECUTION COMPLETE')
        print('█' * 70)
        print(f'\n✓ Detection ensemble trained and saved')
        print(f'✓ Models saved to: {MODEL_DIR}/')
        print(f'✓ Evaluation plots saved to: outputs/')
        print(f'\n[NEXT STEP] Train RL agent:')
        print(f'  python main.py --mode train_rl\n')
        
    except KeyboardInterrupt:
        print('\n\n[!] Pipeline interrupted by user')
        sys.exit(1)
    except Exception as e:
        print('\n\n[ERROR] Pipeline failed!')
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()