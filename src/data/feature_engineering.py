"""
Etape 2 — Feature engineering temporel (avant le split)
"""

import pandas as pd

# Colonnes physiques primaires : on ajoute lags + rolling
COLS_WITH_LAGS = [
    "voltage_v",
    "frequency_hz",
    "power_w",
    "thd_percent",
    "voltage_out_of_tolerance",
    "freq_out_of_tolerance",
]

WINDOW = 10  # 10 cycles = 20 secondes d'historique


def add_window_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute pour chaque colonne de COLS_WITH_LAGS :
      - 5 features de decalage : col_lag1 ... col_lag5
      - 1 moyenne glissante    : col_roll_mean (sur 10 cycles precedents)
      - 1 ecart-type glissant  : col_roll_std  (mesure d'instabilite)

    REGLE CAUSALE : shift(1) applique avant rolling => la feature
    a l'instant t utilise uniquement les valeurs t-1, t-2, ..., t-10.
    Jamais la valeur courante t. Ceci evite toute fuite d'information.
    """
    for col in COLS_WITH_LAGS:
        for lag in range(1, 6):
            # lag k => valeur k cycles (2*k secondes) avant t
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
        # shift(1) garantit que t n'est pas inclus dans la fenetre
        df[f"{col}_roll_mean"] = df[col].shift(1).rolling(WINDOW).mean()
        df[f"{col}_roll_std"] = df[col].shift(1).rolling(WINDOW).std()
    return df


def build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline complet de feature engineering temporel.
    
    Args:
        df: DataFrame trié chronologiquement
        
    Returns:
        DataFrame enrichi des features temporelles, sans NaN
    """
    print("\n" + "=" * 60)
    print("ETAPE 2 — Feature engineering temporel (avant le split)")
    print("=" * 60)

    df_feat = add_window_features(df.copy())

    # Suppression des lignes sans historique suffisant (10 premieres)
    n_avant = len(df_feat)
    df_feat = df_feat.dropna().reset_index(drop=True)
    n_apres = len(df_feat)

    print(f"  Lignes supprimees (manque d'historique) : {n_avant - n_apres}")
    print(f"  Lignes restantes                        : {n_apres:,}")

    # Nouvelles features generees
    n_lags = len(COLS_WITH_LAGS) * 5   # 6 colonnes x 5 lags
    n_rolling = len(COLS_WITH_LAGS) * 2  # 6 colonnes x 2 statistiques
    print(f"  Features ajoutees : {n_lags} lags + {n_rolling} rolling = {n_lags + n_rolling} nouvelles colonnes")

    return df_feat