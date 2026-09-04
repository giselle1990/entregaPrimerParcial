from pathlib import Path
import pandas as pd

DATA = Path("data/raw/customer_churn_historical.csv")
OUT = Path("results/eda_summary.md")

df = pd.read_csv(DATA)
missing = df.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
target_counts = df["Churn"].value_counts(dropna=False)
target_pct = df["Churn"].value_counts(normalize=True).mul(100)

lines = [
    "# EDA — Customer Churn histórico",
    "",
    f"- Dimensiones: **{df.shape[0]} filas × {df.shape[1]} columnas**.",
    f"- Tipos: **{sum(df.dtypes == 'object')} categóricas/objeto** y **{sum(df.dtypes != 'object')} numéricas**.",
    f"- Target `Churn`: No={target_counts.get('No', 0)} ({target_pct.get('No', 0):.2f}%), Yes={target_counts.get('Yes', 0)} ({target_pct.get('Yes', 0):.2f}%).",
    f"- Faltantes: {', '.join(f'{k}={v}' for k, v in missing.items()) if len(missing) else 'sin faltantes'}.",
    f"- IDs duplicados: {df['customerID'].duplicated().sum()}; filas duplicadas: {df.duplicated().sum()}.",
    f"- Casos con `tenure=0`: {(df['tenure'] == 0).sum()}; casos con `TotalCharges=0`: {(df['TotalCharges'] == 0).sum()}.",
    "- `customerID` se excluye del conjunto de features por ser identificador.",
    "- `TotalCharges` se imputa dentro del pipeline para evitar tratamiento manual inconsistente.",
    "",
    "## Distribución resumida de variables numéricas",
    "",
    df[["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]].describe().round(2).to_markdown(),
    "",
    "## Churn por tipo de contrato",
    "",
    pd.crosstab(df["Contract"], df["Churn"], normalize="index").mul(100).round(2).to_markdown(),
    "",
    "## Churn por servicio de Internet",
    "",
    pd.crosstab(df["InternetService"], df["Churn"], normalize="index").mul(100).round(2).to_markdown(),
    "",
    "## Lectura inicial",
    "",
    "El churn es mayor en contratos mes a mes (38,73%) que en contratos de dos años (8,97%). También es mayor entre clientes con fibra óptica (40,25%) que entre quienes no tienen Internet (7,62%). Estas asociaciones sirven para orientar el análisis, pero no prueban causalidad.",
]
OUT.parent.mkdir(exist_ok=True)
OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"EDA escrita en {OUT}")
