"""
Etape 6 — Application de SMOTE (train uniquement)
"""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE


def apply_smote_A(X_train_scaled, y_train_A, sampling_strategy=0.10, k_neighbors=3, random_state=42):
    """
    Applique SMOTE pour le Modele A (pre_fault).
    
    Ratio 1:10 car seulement 92 vrais positifs.
    k_neighbors=3 car tres peu de positifs reels.
    """
    print("\n" + "=" * 60)
    print("ETAPE 6 — Application de SMOTE (train uniquement)")
    print("=" * 60)

    smote_A = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=random_state
    )
    X_res_A, y_res_A = smote_A.fit_resample(X_train_scaled, y_train_A)

    n_synth_A = int((y_res_A == 1).sum()) - int(y_train_A.sum())
    print(f"\n  [Modele A — pre_fault]")
    print(f"    Avant SMOTE  : {int(y_train_A.sum()):,} positifs | {int((y_train_A==0).sum()):,} negatifs")
    print(f"    Apres SMOTE  : {int((y_res_A==1).sum()):,} positifs | {int((y_res_A==0).sum()):,} negatifs")
    print(f"    Cas synthetiques generes : {n_synth_A:,}")
    print(f"    Ratio final  : 1:{int((y_res_A==0).sum()//(y_res_A==1).sum())}")
    print(f"    Total lignes train A : {len(y_res_A):,}")

    return X_res_A, y_res_A


def apply_smote_B(X_train_scaled, y_train_B, sampling_strategy=0.33, k_neighbors=5, random_state=42):
    """
    Applique SMOTE pour le Modele B (outage).
    
    Ratio 1:3 car 4 491 vrais positifs (desequilibre moins severe).
    """
    smote_B = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=random_state
    )
    X_res_B, y_res_B = smote_B.fit_resample(X_train_scaled, y_train_B)

    n_synth_B = int((y_res_B == 1).sum()) - int(y_train_B.sum())
    print(f"\n  [Modele B — outage]")
    print(f"    Avant SMOTE  : {int(y_train_B.sum()):,} positifs | {int((y_train_B==0).sum()):,} negatifs")
    print(f"    Apres SMOTE  : {int((y_res_B==1).sum()):,} positifs | {int((y_res_B==0).sum()):,} negatifs")
    print(f"    Cas synthetiques generes : {n_synth_B:,}")
    print(f"    Ratio final  : 1:{int((y_res_B==0).sum()//(y_res_B==1).sum())}")
    print(f"    Total lignes train B : {len(y_res_B):,}")

    return X_res_B, y_res_B