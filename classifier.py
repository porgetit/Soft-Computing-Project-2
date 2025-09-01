import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime

from mlxtend.classifier import Adaline
from mlxtend.plotting import plot_decision_regions


DATA_PATH = "heart_predictions.csv"
FEATURES = ["age", "sex", "trestbps", "chol", "hdl", "fbs"]
CONT_FEATURES = ["age", "trestbps", "chol", "hdl"]  # estandarizar
TARGET = "fis_pred"  # objetivo a imitar (producido por el FIS)


def load_dataset(path: str):
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Faltan columnas en {path}: {missing}. Genera primero {DATA_PATH} con el FIS."
        )
    X = df[FEATURES].astype(float).values
    y = df[TARGET].astype(int).values
    return df, X, y


def standardize_selected(X: np.ndarray, cols: list[str], all_cols: list[str]):
    X_std = X.copy().astype(float)
    means = {}
    stds = {}
    for c in cols:
        if c not in all_cols:
            continue
        j = all_cols.index(c)
        mu = X_std[:, j].mean()
        sd = X_std[:, j].std() + 1e-12
        X_std[:, j] = (X_std[:, j] - mu) / sd
        means[c] = mu
        stds[c] = sd
    return X_std, means, stds


def evaluate(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    acc = float((y_true == y_pred).mean())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return acc, tp, fp, fn, tn


def main():
    print("[INFO] Cargando dataset y etiquetas del FIS…")
    df, X, y = load_dataset(DATA_PATH)
    print(f"[INFO] {DATA_PATH} cargado: {X.shape[0]} filas, {X.shape[1]} features.")

    # Estandarizar solo continuas (age, trestbps, chol, hdl)
    X_std, means, stds = standardize_selected(X, CONT_FEATURES, FEATURES)
    print("[INFO] Estandarización aplicada a:", ", ".join(means.keys()))

    # Entrenar Adaline para imitar al FIS
    ada = Adaline(
        eta=0.01,
        epochs=16,
        minibatches=5,
        random_seed=1,
        print_progress=3,
    )
    ada.fit(X_std, y)

    # Evaluación de concordancia con el FIS
    y_pred = ada.predict(X_std)
    acc, tp, fp, fn, tn = evaluate(y, y_pred)
    print("\n[RESULTADOS] Imitación del FIS (en el mismo conjunto):")
    print(f"- Accuracy: {acc:.4f}")
    print("- Matriz de confusión (verdadero x predicho):")
    print(f"    TN={tn}  FP={fp}")
    print(f"    FN={fn}  TP={tp}")

    # Pesos y bias del modelo (escala estandarizada)
    w = np.asarray(ada.w_).ravel()
    b = float(np.asarray(ada.b_).ravel()[0])
    print("\n[MODELO] Parámetros aprendidos (escala estandarizada):")
    for name, wi in zip(FEATURES, w):
        print(f"- w[{name}] = {wi:.6f}")
    print(f"- b (bias) = {b:.6f}")

    # Reescalar a la escala original de las variables continuas
    w_orig = w.copy()
    b_orig = b
    for c in CONT_FEATURES:
        j = FEATURES.index(c)
        sd = stds[c]
        mu = means[c]
        w_orig[j] = w[j] / sd
        b_orig -= w[j] * (mu / sd)
    print("\n[MODELO] Coeficientes en escala original:")
    for name, wi in zip(FEATURES, w_orig):
        print(f"- w_orig[{name}] = {wi:.6f}")
    print(f"- b_orig (bias) = {b_orig:.6f}")

    # Guardar parámetros a disco
    std_weights = {name: float(val) for name, val in zip(FEATURES, w)}
    orig_weights = {name: float(val) for name, val in zip(FEATURES, w_orig)}
    meta = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "library": "mlxtend",
        "estimator": "Adaline",
        "features": FEATURES,
        "hyperparams": {
            "eta": 0.02,
            "epochs": 50,
            "minibatches": 5,
            "random_seed": 1,
        },
        "metrics": {
            "accuracy": acc,
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        },
        "standardization": {
            "standardized_columns": CONT_FEATURES,
            "means": {k: float(v) for k, v in means.items()},
            "stds": {k: float(v) for k, v in stds.items()},
        },
        "scale_standardized": {
            "weights": std_weights,
            "bias": float(b),
        },
        "scale_original": {
            "weights": orig_weights,
            "bias": float(b_orig),
        },
    }

    with open("adaline_params.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    params_df = pd.DataFrame(
        {
            "feature": FEATURES + ["__bias__"],
            "w_std": [std_weights[f] for f in FEATURES] + [float(b)],
            "w_orig": [orig_weights[f] for f in FEATURES] + [float(b_orig)],
        }
    )
    params_df.to_csv("adaline_params.csv", index=False)
    print("\n[OK] Guardados parámetros en 'adaline_params.json' y 'adaline_params.csv'.")

    # Curva de costo de entrenamiento
    plt.figure(figsize=(5, 3))
    plt.plot(range(len(ada.cost_)), ada.cost_, lw=2)
    plt.xlabel("Iteraciones")
    plt.ylabel("Coste")
    plt.title("Adaline: evolución del coste")
    plt.tight_layout()
    plt.show()

    # Visualización 2D opcional con dos features continuas
    vis_cols = [c for c in ("trestbps", "chol", "age", "hdl") if c in FEATURES][:2]
    if len(vis_cols) == 2:
        i, j = FEATURES.index(vis_cols[0]), FEATURES.index(vis_cols[1])
        X_vis = X_std[:, [i, j]]

        ada2d = Adaline(
            eta=0.02,
            epochs=50,
            minibatches=5,
            random_seed=1,
            print_progress=0,
        )
        ada2d.fit(X_vis, y)

        plt.figure(figsize=(5, 4))
        plot_decision_regions(X_vis, y, clf=ada2d, legend=2)
        plt.xlabel(f"{vis_cols[0]} (std)")
        plt.ylabel(f"{vis_cols[1]} (std)")
        plt.title("Regiones de decisión (2D)")
        plt.tight_layout()
        plt.show()
    else:
        print("[AVISO] No hay suficientes columnas continuas para una visualización 2D.")


if __name__ == "__main__":
    main()
