"""
Etape 8 — Rapport de verification finale
"""

import pandas as pd


def verify_and_export(
    X_res_A, y_res_A,
    X_res_B, y_res_B,
    X_test_scaled, y_test_A, y_test_B,
    features, test,
    y_train_A, y_train_B
):
    """
    Exporte les 4 fichiers CSV et verifie la coherence.
    """
    print("\n" + "=" * 60)
    print("ETAPE 7 — Export des fichiers CSV")
    print("=" * 60)

    # --- Train A (pre_fault) ---
    df_train_A = pd.DataFrame(X_res_A, columns=features)
    df_train_A["target"] = y_res_A
    df_train_A.to_csv("train_pre_fault.csv", index=False)

    # --- Test A (pre_fault) — avec timestamp ---
    df_test_A = pd.DataFrame(X_test_scaled, columns=features)
    df_test_A.insert(0, "timestamp", test["timestamp"].values)
    df_test_A["target"] = y_test_A
    df_test_A.to_csv("test_pre_fault.csv", index=False)

    # --- Train B (outage) ---
    df_train_B = pd.DataFrame(X_res_B, columns=features)
    df_train_B["target"] = y_res_B
    df_train_B.to_csv("train_outage.csv", index=False)

    # --- Test B (outage) — avec timestamp ---
    df_test_B = pd.DataFrame(X_test_scaled, columns=features)
    df_test_B.insert(0, "timestamp", test["timestamp"].values)
    df_test_B["target"] = y_test_B
    df_test_B.to_csv("test_outage.csv", index=False)

    # ================================================================
    # RAPPORT DE VERIFICATION
    # ================================================================
    print("\n" + "=" * 60)
    print("RAPPORT DE VERIFICATION FINALE")
    print("=" * 60)

    fichiers = {
        "train_pre_fault.csv": df_train_A,
        "test_pre_fault.csv": df_test_A,
        "train_outage.csv": df_train_B,
        "test_outage.csv": df_test_B,
    }

    for nom, df_ in fichiers.items():
        t = df_["target"].value_counts().to_dict()
        print(f"  {nom:30s} | {len(df_):>7,} lignes | {df_.shape[1]:>3} colonnes | 0={t.get(0,0):,}  1={t.get(1,0):,}")

    print("\n  Verifications de coherence :")

    # Aucune fuite : le test n'a pas ete touche par SMOTE
    assert len(df_test_A) == len(test), "ERREUR : test A modifie !"
    assert len(df_test_B) == len(test), "ERREUR : test B modifie !"
    print("  [OK] Test sets non modifies par SMOTE")

    # Les negatifs du train ne changent pas apres SMOTE
    assert int((y_res_A == 0).sum()) == int((y_train_A == 0).sum()), "ERREUR : negatifs modifies A"
    assert int((y_res_B == 0).sum()) == int((y_train_B == 0).sum()), "ERREUR : negatifs modifies B"
    print("  [OK] SMOTE n'a que sur-echantillonne (negatifs inchanges)")

    # Pas de valeurs manquantes
    assert df_train_A.isnull().sum().sum() == 0
    assert df_train_B.isnull().sum().sum() == 0
    assert df_test_A.isnull().sum().sum() == 0
    assert df_test_B.isnull().sum().sum() == 0
    print("  [OK] Aucune valeur manquante dans les 4 fichiers")

    print("\n4 fichiers CSV generes avec succes.")
    print("  train_pre_fault.csv")
    print("  test_pre_fault.csv")
    print("  train_outage.csv")
    print("  test_outage.csv")