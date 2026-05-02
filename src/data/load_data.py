"""
Etape 1 — Chargement et tri chronologique
"""

import pandas as pd


def load_and_sort_data(filepath: str) -> pd.DataFrame:
    """
    Charge le dataset, parse les timestamps, trie chronologiquement.
    
    Args:
        filepath: /home/matchabo/Bureau/prediction_pannes_electriques/data/raw/dataset_electricity_outage.csv
        
    Returns:
        DataFrame trié par timestamp, index reset
    """
    print("=" * 60)
    print("ETAPE 1 — Chargement et tri chronologique")
    print("=" * 60)

    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"  Dataset charge     : {len(df):,} lignes x {df.shape[1]} colonnes")
    print(f"  Plage temporelle   : {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(f"  Valeurs manquantes : {df.isnull().sum().sum()}")

    return df