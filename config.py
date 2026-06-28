# config.py
# ─────────────────────────────────────────────────────────────
# Central configuration for I²DARS.
# All hyperparameters, paths, and mappings are defined here.
# Updated for CSE-CIC-IDS2018 dataset
# ─────────────────────────────────────────────────────────────

import os

# ── Reproducibility ──────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.20
VAL_SIZE     = 0.10

# ── CSE-CIC-IDS2018 Attack Category Mapping ──────────────────
# Map individual attack labels to broad categories
ATTACK_CATEGORIES = {
    'benign'                    : 0,
    
    # DoS/DDoS attacks
    'dos attacks-hulk'          : 1,
    'dos attacks-slowhttptest'  : 1,
    'dos attacks-slowloris'     : 1,
    'dos attacks-goldeneye'     : 1,
    'ddos attacks-loit'         : 1,
    'ddos attacks-hoic'         : 1,
    
    # Port Scan / Network Recon
    'reconnaissance-portscan'   : 2,
    'reconnaissance-ping sweep' : 2,
    
    # Brute Force
    'brute force -web'          : 3,
    'brute force -xss'          : 3,
    'brute force-ftp'           : 3,
    'brute force-ssh'           : 3,
    
    # Infiltration / Privilege Escalation
    'infiltration'              : 4,
    'botnet'                    : 4,
    'web attack-sql injection'  : 4,
    'web attack-xss'            : 4,
}

ATTACK_NAMES = {
    0 : 'BENIGN',
    1 : 'DOS_DDOS',
    2 : 'PORT_SCAN',
    3 : 'BRUTE_FORCE',
    4 : 'INFILTRATION',
}

NUM_CLASSES = len(ATTACK_NAMES)

# ── MITRE ATT&CK Mapping ─────────────────────────────────────
MITRE_MAPPING = {
    'DOS_DDOS': {
        'tactic'        : 'Impact',
        'technique_id'  : 'T1499',
        'technique_name': 'Endpoint Denial of Service',
        'key_features'  : ['Total Fwd Packets', 'Total Backward Packets',
                           'Total Length of Fwd Packets', 'Fwd Packet Length Mean'],
    },
    'PORT_SCAN': {
        'tactic'        : 'Discovery',
        'technique_id'  : 'T1046',
        'technique_name': 'Network Service Discovery',
        'key_features'  : ['Destination Port', 'Protocol',
                           'Flow Duration', 'Total Fwd Packets'],
    },
    'BRUTE_FORCE': {
        'tactic'        : 'Credential Access',
        'technique_id'  : 'T1110',
        'technique_name': 'Brute Force',
        'key_features'  : ['Source Port', 'Destination Port',
                           'Fwd Packet Length Mean', 'Bwd Packet Length Mean'],
    },
    'INFILTRATION': {
        'tactic'        : 'Privilege Escalation',
        'technique_id'  : 'T1068',
        'technique_name': 'Exploitation for Privilege Escalation',
        'key_features'  : ['Flow Duration', 'Total Fwd Packets',
                           'Total Bwd Packets', 'Total Length of Fwd Packets'],
    },
}

# ── RL / CMDP Configuration ──────────────────────────────────
ACTIONS = [
    'block_ip',
    'rate_limit',
    'isolate_host',
    'honeypot_redirect',
    'alert_only',
]

ACTION_COSTS = {
    'block_ip'          : 0.30,
    'rate_limit'        : 0.10,
    'isolate_host'      : 0.80,
    'honeypot_redirect' : 0.20,
    'alert_only'        : 0.00,
}

COST_BUDGET     = 0.50
LAMBDA_COST     = 0.50

PROTECTED_RESOURCES = [
    'database_server',
    'payment_gateway',
    'hr_system',
    'domain_controller',
]

# RL hyperparameters
RL_EPISODES      = 5000
LEARNING_RATE    = 0.10
DISCOUNT_FACTOR  = 0.95
EPSILON_START    = 1.00
EPSILON_END      = 0.01
EPSILON_DECAY    = 0.995

# ── File Paths ───────────────────────────────────────────────
MODEL_DIR          = 'saved_models'
RF_MODEL_PATH      = os.path.join(MODEL_DIR, 'rf_model.pkl')
XGB_MODEL_PATH     = os.path.join(MODEL_DIR, 'xgb_model.json')
DNN_MODEL_PATH     = os.path.join(MODEL_DIR, 'dnn_model.h5')
CATBOOST_MODEL_PATH= os.path.join(MODEL_DIR, 'catboost_model.cbm')
RL_QTABLE_PATH     = os.path.join(MODEL_DIR, 'rl_agent.pkl')
SCALER_PATH        = os.path.join(MODEL_DIR, 'scaler.pkl')
ENCODER_PATH       = os.path.join(MODEL_DIR, 'encoders.pkl')
FEATURE_COLS_PATH  = os.path.join(MODEL_DIR, 'feature_cols.json')

# ── Dataset Paths (CSE-CIC-IDS2018) ──────────────────────────
DATA_DIR = os.path.join('data', 'CSE-CIC-IDS2018')

# Files to use (UPDATE WITH YOUR ACTUAL FILENAMES)
# These match the exact filenames in your data folder
DATA_FILES = [
    'Friday-02-03-2018_TrafficForML_CICFlowMeter.csv',
    'Friday-16-02-2018_TrafficForML_CICFlowMeter.csv',
    'Friday-23-02-2018_TrafficForML_CICFlowMeter.csv',
    'Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv',
    'Thursday-20-02-2018_TrafficForML_CICFlowMeter.csv',
]

# All CSE-CIC-IDS2018 CSV files available (if you have more)
ALL_DATA_FILES = [
    'Friday-02-03-2018_TrafficForML_CICFlowMeter.csv',
    'Friday-16-02-2018_TrafficForML_CICFlowMeter.csv',
    'Friday-23-02-2018_TrafficForML_CICFlowMeter.csv',
    'Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv',
    'Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv',
    'Thursday-20-02-2018_TrafficForML_CICFlowMeter.csv',
    'Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv',
    'Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv',
    'Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv',
    'Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv',
]