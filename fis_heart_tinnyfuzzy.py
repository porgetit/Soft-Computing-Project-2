# fis_heart_tinyfuzzy.py
# FIS Sugeno (orden 0) para heart.csv usando tinyfuzzy
# Variables (Framingham ∩ CSV): age, sex, trestbps (SBP), chol, hdl, fbs
# Salidas: heart_predictions.csv (fis_score, fis_pred) y fis_breaks_used.txt

import pandas as pd
# import numpy as np
import sys

# Asegúrate de tener tinyfuzzy.py en el mismo directorio o en el PYTHONPATH
import tinyfuzzy as tf
FuzzySet = tf.FuzzySet
LinguisticVariable = tf.LinguisticVariable
trimf, trapmf = tf.trimf, tf.trapmf
sugeno = tf.sugeno

# ------------------- Configuración -------------------
PATH_CSV = "heart.csv"
OUT_PRED = "heart_predictions.csv"
OUT_BREAKS = "fis_breaks_used.txt"
THRESHOLD = 0.5  # decisión final

# ------------------- Utilidades -------------------
def qbreaks(s: pd.Series):
    """Cuantiles robustos para definir low/mid/high a partir del propio CSV."""
    s = s.astype(float)
    q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_dict()
    return {
        "min": float(s.min()), "q05": q[0.05], "q25": q[0.25],
        "q50": q[0.5], "q75": q[0.75], "q95": q[0.95], "max": float(s.max())
    }

def make_continuous_lv(name, series, protective=False):
    """
    Crea una variable lingüística con etiquetas low/mid/high:
    - low: trapecio en cola baja (min, q05, q25, q50)
    - mid: triángulo (q25, q50, q75)
    - high: trapecio en cola alta (q50, q75, q95, max)
    Nota: 'protective' no invierte membresías; se refleja en las reglas.
    """
    b = qbreaks(series)

    # Construimos puntos en orden y forzamos estrictamente creciente
    def inc4(vals):
        vals = list(vals)
        for i in range(1, len(vals)):
            if vals[i] <= vals[i-1]:
                vals[i] = vals[i-1] + 1e-6
        return vals

    a1, a2, a3, a4 = inc4([b["min"], b["q05"], b["q25"], b["q50"]])
    h1, h2, h3, h4 = inc4([b["q50"], b["q75"], b["q95"], b["max"]])
    m1, m2, m3 = b["q25"], b["q50"], b["q75"]
    # Evitar triángulo degenerado
    if not (m1 < m2 < m3):
        # Perturbar ligeramente si hay empates
        m1, m2, m3 = float(m1), float(m2), float(m3)
        if not m1 < m2: m2 = m1 + 1e-6
        if not m2 < m3: m3 = m2 + 1e-6

    return LinguisticVariable(
        name=name,
        universe=(float(b["min"]), float(b["max"])),
        sets={
            "low":  FuzzySet("low",  lambda xs, A=a1,B=a2,C=a3,D=a4: trapmf(xs, A,B,C,D)),
            "mid":  FuzzySet("mid",  lambda xs, A=m1,B=m2,C=m3:   trimf(xs, A,B,C)),
            "high": FuzzySet("high", lambda xs, A=h1,B=h2,C=h3,D=h4: trapmf(xs, A,B,C,D)),
        }
    )

def crisp0(xs):  # para binarias "0"
    return trapmf(xs, -0.5, -0.1, 0.1, 0.5)

def crisp1(xs):  # para binarias "1"
    return trapmf(xs,  0.5,  0.9, 1.1, 1.5)

def AND(var, setname):  # helper para antecedente
    return (var, setname, "AND")

def LAST(var, setname):  # último literal del antecedente
    return (var, setname, "")

# ------------------- Cargar datos -------------------
df = pd.read_csv(PATH_CSV)

# Columnas esperadas (Framingham ∩ CSV)
needed = ["age", "sex", "trestbps", "chol", "hdl", "fbs"]
missing = [c for c in needed if c not in df.columns]
if missing:
    print(f"[AVISO] Faltan columnas en {PATH_CSV}: {missing}. Se continuará con las disponibles.", file=sys.stderr)

# ------------------- Variables lingüísticas -------------------
variables = {}

if "age" in df.columns:
    variables["age"] = make_continuous_lv("age", df["age"])

if "trestbps" in df.columns:
    # Usaremos el nombre 'sbp' internamente
    variables["sbp"] = make_continuous_lv("sbp", df["trestbps"])

if "chol" in df.columns:
    variables["chol"] = make_continuous_lv("chol", df["chol"])

if "hdl" in df.columns:
    # HDL alto es protector (se reflejará en las reglas)
    variables["hdl"] = make_continuous_lv("hdl", df["hdl"], protective=True)

if "sex" in df.columns:
    variables["sex"] = LinguisticVariable(
        "sex", (0.0, 1.0),
        {
            "female": FuzzySet("female", lambda xs: crisp0(xs)),
            "male":   FuzzySet("male",   lambda xs: crisp1(xs)),
        }
    )

if "fbs" in df.columns:
    variables["fbs"] = LinguisticVariable(
        "fbs", (0.0, 1.0),
        {
            "no":  FuzzySet("no",  lambda xs: crisp0(xs)),
            "yes": FuzzySet("yes", lambda xs: crisp1(xs)),
        }
    )

if not variables:
    raise RuntimeError("No hay variables disponibles para construir el FIS.")

# ------------------- Base de reglas (Sugeno, orden 0) -------------------
rules = []

# Riesgo alto
if all(v in variables for v in ["age", "sbp", "chol"]):
    rules.append(([AND("age","high"), AND("sbp","high"), LAST("chol","high")], 0.90))
    rules.append(([AND("age","high"), AND("sbp","high"), LAST("chol","mid")],  0.80))

if all(v in variables for v in ["fbs","sbp"]):
    rules.append(([AND("fbs","yes"), LAST("sbp","high")], 0.85))

if all(v in variables for v in ["sex","age","chol"]):
    rules.append(([AND("sex","male"), AND("age","high"), LAST("chol","high")], 0.80))
    rules.append(([AND("sex","male"), AND("age","high"), LAST("chol","mid")],  0.70))

if all(v in variables for v in ["chol","fbs"]):
    rules.append(([AND("chol","high"), LAST("fbs","yes")], 0.75))

if all(v in variables for v in ["sbp","chol"]):
    rules.append(([AND("sbp","high"), LAST("chol","mid")], 0.65))

if all(v in variables for v in ["age","fbs"]):
    rules.append(([AND("age","mid"), LAST("fbs","yes")], 0.60))

if all(v in variables for v in ["age","sbp"]):
    rules.append(([AND("age","mid"), LAST("sbp","high")], 0.60))

# HDL como factor protector
if all(v in variables for v in ["hdl","chol"]):
    rules.append(([AND("hdl","low"), LAST("chol","high")], 0.75))
    rules.append(([AND("hdl","high"), LAST("chol","low")], 0.20))

if all(v in variables for v in ["hdl","sbp"]):
    rules.append(([AND("hdl","low"), LAST("sbp","high")], 0.70))
    rules.append(([AND("hdl","high"), LAST("sbp","low")], 0.25))

# Riesgo bajo
if all(v in variables for v in ["age","sbp","chol","fbs","hdl"]):
    rules.append(([AND("age","low"), AND("sbp","low"), AND("chol","low"), AND("hdl","high"), LAST("fbs","no")], 0.10))

if all(v in variables for v in ["sex","age","hdl"]):
    rules.append(([AND("sex","female"), AND("age","low"), LAST("hdl","high")], 0.15))

if all(v in variables for v in ["age","sbp","chol"]):
    rules.append(([AND("age","mid"), AND("sbp","low"), LAST("chol","low")], 0.30))

# Fallback (por si hubiera poquísimas columnas)
if len(rules) == 0:
    first_var = next(iter(variables.keys()))
    first_set = list(variables[first_var].sets.keys())[0]
    rules.append(([LAST(first_var, first_set)], 0.5))

# ------------------- Evaluación sobre las 303 filas -------------------
def eval_row(row):
    crisp = {}
    if "age"       in df.columns: crisp["age"]  = float(row["age"])
    if "sex"       in df.columns: crisp["sex"]  = float(row["sex"])
    if "trestbps"  in df.columns: crisp["sbp"]  = float(row["trestbps"])
    if "chol"      in df.columns: crisp["chol"] = float(row["chol"])
    if "hdl"       in df.columns: crisp["hdl"]  = float(row["hdl"])
    if "fbs"       in df.columns: crisp["fbs"]  = float(row["fbs"])
    return sugeno(crisp, variables, rules)

scores = df.apply(eval_row, axis=1)
pred = (scores >= THRESHOLD).astype(int)

# ------------------- Guardado de resultados -------------------
use_cols = [c for c in ["age","sex","trestbps","chol","hdl","fbs"] if c in df.columns]
out = df.copy()
out["fis_score"] = scores.values
out["fis_pred"]  = pred.values
out.to_csv(OUT_PRED, index=False)

# Guardar cuantiles usados
def save_breaks(name, series):
    b = qbreaks(series)
    return (f"{name}: min={b['min']:.3f}, q05={b['q05']:.3f}, q25={b['q25']:.3f}, "
            f"q50={b['q50']:.3f}, q75={b['q75']:.3f}, q95={b['q95']:.3f}, max={b['max']:.3f}")

lines = []
for c in ["age","trestbps","chol","hdl"]:
    if c in df.columns:
        lines.append(save_breaks(c, df[c]))
with open(OUT_BREAKS, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

# Resumen por consola
n1 = int(pred.sum())
n0 = int((1 - pred).sum())
print(f"[OK] Generado {OUT_PRED} con {len(out)} filas.")
print(f"    Umbral = {THRESHOLD} → pred_1={n1}, pred_0={n0}")
print(f"[OK] Guardado {OUT_BREAKS} con cortes (cuantiles) usados.")
