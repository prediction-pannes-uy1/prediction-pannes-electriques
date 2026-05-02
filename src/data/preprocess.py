"""
Etape 3 — Split temporel strict 80/20
Etape 4 — Definition des features
Etape 5 — Normalisation StandardScaler
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Colonnes exclues des features
COLS_EXCLUES = [
    "timestamp",
    "cycle",
    "cycle_global",
    "outage",
    "micro_outage",
    "pre_fault",
]


def train_test_split_temporal(df_feat: pd.DataFrame, test_size: float = 0.20):
    """
    Split temporel strict sans melange aleatoire.
    
    POURQUOI pas de melange aleatoire ?
    Melanger briserait la chronologie. Le modele verrait des donnees
    du futur pendant l'entrainement => performances artificiellement
    gonflees en test, mauvaises en production.
    
    Args:
        df_feat: DataFrame avec features temporelles
        test_size: proportion du test (defaut: 0.20)
        
    Returns:
        train, test: DataFrames
    """
    print("\n" + "=" * 60)
    print("ETAPE 3 — Split temporel strict 80/20")
    print("=" * 60)

    split = int(len(df_feat) * (1 - test_size))
    train = df_feat.iloc[:split].copy()
    test = df_feat.iloc[split:].copy()

    print(f"  Train : {len(train):,} lignes  ({train['timestamp'].min()} -> {train['timestamp'].max()})")
    print(f"  Test  : {len(test):,} lignes  ({test['timestamp'].min()} -> {test['timestamp'].max()})")

    return train, test


def get_feature_list(df_columns) -> list:
    """
    Retourne la liste des features en excluant les colonnes non pertinentes.
    
    Colonnes exclues :
      - timestamp      : identifiant temporel (pas une mesure)
      - cycle          : compteur local reinitialise (identifiant)
      - cycle_global   : numero de ligne global (identifiant)
      - outage         : CIBLE du modele B
      - micro_outage   : exclue pour eviter fuite (co-occurrence avec pre_fault/outage)
      - pre_fault      : CIBLE du modele A
    """
    print("\n" + "=" * 60)
    print("ETAPE 4 — Definition des features")
    print("=" * 60)

    FEATURES = [c for c in df_columns if c not in COLS_EXCLUES]

    print(f"  Nombre total de features : {len(FEATURES)}")
    print(f"  Colonnes exclues         : {COLS_EXCLUES}")
    print("\n  Detail des features par categorie :")
    print("  [Mesures physiques]        :", [f for f in FEATURES if f in
          ["voltage_v", "frequency_hz", "power_w", "thd_percent"]])
    print("  [Ecarts a la norme]        :", [f for f in FEATURES if "deviation" in f and "lag" not in f and "roll" not in f])
    print("  [Flags tolerance]          :", [f for f in FEATURES if "tolerance" in f and "lag" not in f and "roll" not in f])
    print("  [Contexte temporel]        :", [f for f in FEATURES if f in
          ["hour", "is_peak_hour", "is_night"]])
    print("  [Memoire reseau]           :", [f for f in FEATURES if f in
          ["outage_count", "micro_outage_count", "cycles_since_last_outage"]])
    print(f"  [Lags + rolling]           : {len([f for f in FEATURES if 'lag' in f or 'roll' in f])} colonnes")

    return FEATURES


def normalize_data(train, test, features):
    """
    Normalise les donnees avec StandardScaler.
    
    REGLE ANTI-FUITE : scaler.fit() uniquement sur le train.
    """
    print("\n" + "=" * 60)
    print("ETAPE 5 — Normalisation StandardScaler (fit sur train uniquement)")
    print("=" * 60)

    scaler = StandardScaler()

    X_train_raw = train[features].values
    X_test_raw = test[features].values

    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    print(f"  Scaler ajuste sur : {len(X_train_raw):,} lignes du train")
    print(f"  Applique sur test : {len(X_test_raw):,} lignes")
    print(f"  Verification — moyenne train apres scaling : {X_train_scaled.mean():.6f} (attendu : ~0)")
    print(f"  Verification — std train apres scaling     : {X_train_scaled.std():.6f} (attendu : ~1)")

    return X_train_scaled, X_test_scaled, scaler, features


def get_labels(train, test):
    """
    Extrait les labels pour les deux modeles.
    """
    y_train_A = train["pre_fault"].values
    y_test_A = test["pre_fault"].values
    y_train_B = train["outage"].values
    y_test_B = test["outage"].values

    print(f"\n  Distribution classes AVANT SMOTE :")
    print(f"    pre_fault train : 0={int((y_train_A==0).sum()):,}  1={int((y_train_A==1).sum()):,}  ratio=1:{int((y_train_A==0).sum()/max(y_train_A.sum(),1))}")
    print(f"    outage    train : 0={int((y_train_B==0).sum()):,}  1={int((y_train_B==1).sum()):,}  ratio=1:{int((y_train_B==0).sum()/max(y_train_B.sum(),1))}")

    return y_train_A, y_test_A, y_train_B, y_test_B