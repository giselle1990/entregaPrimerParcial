# Primer Parcial — Laboratorio de Minería de Datos
## Proyecto reproducible: Git + DVC + MLflow + Model Registry

**Caso:** predicción de Customer Churn en telecomunicaciones.  
**Entrega:** Primer Parcial / Entrega 1.  
**Objetivo:** transformar el dataset histórico y el análisis exploratorio en un proyecto Python reproducible, versionado y trazable, finalizando con un modelo candidato registrable en MLflow Model Registry.

## 1. Qué incluye esta entrega

- Estructura modular de proyecto Python.
- Dataset histórico preparado para tracking con DVC (el CSV está ignorado por Git).
- EDA reproducible y resumen de calidad de datos.
- Split train/test estratificado y reproducible (`random_state=42`, test=20%).
- `Pipeline` de scikit-learn con imputación, encoding y escalado.
- Exclusión explícita de `customerID` como predictor.
- Seis configuraciones de experimentación: baseline, regresión logística, random forest y ajuste de threshold.
- Registro de parámetros, métricas, tags y modelo en MLflow.
- Selección justificada del candidato y opción de alta en Model Registry.
- Archivos de resultados obtenidos con la misma partición reproducible.
- Scripts para configurar DVC/DagsHub sin guardar secretos en el repositorio.

## 2. Dataset y EDA

Se utiliza `data/raw/customer_churn_historical.csv` (7.043 filas, 21 columnas). El target es `Churn` (`Yes`/`No`). El identificador `customerID` no se utiliza como feature. `TotalCharges` contiene faltantes intencionales y se imputa dentro del pipeline.

La distribución del target es aproximadamente 73,63% `No` y 26,37% `Yes`, por lo que Accuracy no se usa como única métrica. El detalle reproducible está en `results/eda_summary.md` y se genera con:

```bash
python scripts/eda.py
```

> `data/production/customer_churn_current.csv` queda reservado para la etapa final de monitoreo/drift y no se usa para entrenar ni seleccionar el modelo.

## 3. Pipeline de Machine Learning

Flujo implementado:

```text
Raw Data
  → drop customerID
  → train/test split estratificado
  → numéricas: imputación mediana + StandardScaler
  → categóricas: imputación moda + OneHotEncoder(handle_unknown="ignore")
  → estimador
  → probabilidad de Churn
  → threshold
  → predicción
```

El preprocesamiento vive dentro de `sklearn.Pipeline`/`ColumnTransformer`, de modo que entrenamiento e inferencia reutilizan las mismas transformaciones.

## 4. Modelos y experimentos

Se definieron seis Runs relevantes:

| Run | Familia | Threshold |
|---|---|---:|
| `dummy_most_frequent` | Baseline | 0.50 |
| `logreg_c0.5_t0.5` | Regresión logística | 0.50 |
| `logreg_c1_t0.5` | Regresión logística | 0.50 |
| `logreg_c1_t0.35` | Regresión logística + threshold | 0.35 |
| `rf_200_t0.5` | Random Forest | 0.50 |
| `rf_300_depth12_t0.5` | Random Forest | 0.50 |

Métricas: Accuracy, Precision, Recall, F1, ROC-AUC y matriz de confusión (TN/FP/FN/TP).

### Resultados reproducidos

| Run | Precision | Recall | F1 | ROC-AUC | FN |
|---|---:|---:|---:|---:|---:|
| dummy_most_frequent | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 372 |
| logreg_c0.5_t0.5 | 0.6601 | 0.4489 | 0.5344 | 0.8120 | 205 |
| logreg_c1_t0.5 | 0.6640 | 0.4516 | 0.5376 | 0.8120 | 204 |
| **logreg_c1_t0.35** | **0.5507** | **0.6425** | **0.5931** | **0.8120** | **133** |
| rf_200_t0.5 | 0.6413 | 0.3844 | 0.4807 | 0.7906 | 229 |
| rf_300_depth12_t0.5 | 0.5722 | 0.5860 | 0.5790 | 0.8030 | 154 |

## 5. Decisión de negocio y modelo candidato

En churn, un **falso negativo** implica clasificar como estable a un cliente que efectivamente abandonará. Ese error puede impedir que el cliente sea incluido en una acción de retención. Por eso la selección no se guía por Accuracy solamente y se prioriza Recall sin ignorar F1 y ROC-AUC.

El candidato propuesto es **Logistic Regression (`C=1`) con threshold 0,35**. Frente al threshold 0,50, el Recall aumenta de ~0,452 a ~0,642 y los falsos negativos bajan de 204 a 133, manteniendo ROC-AUC ~0,812. El costo es una caída de Precision y un aumento de falsos positivos; ese trade-off se considera razonable para una etapa de detección temprana de riesgo.

El script usa un score de selección explícito:

```text
0.50 × Recall + 0.30 × F1 + 0.20 × ROC-AUC
```

## 6. Instalación

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows PowerShell
pip install -r requirements.txt
```

## 7. DVC + DagsHub

Inicialización / validación:

```bash
dvc init
# Si el archivo ya está preparado, verificarlo:
dvc status
```

Configurar el remote DagsHub sin guardar credenciales en Git:

```bash
export DAGSHUB_DVC_URL="https://dagshub.com/USUARIO/REPOSITORIO.dvc"
bash scripts/configure_dagshub.sh
```

Luego:

```bash
dvc add data/raw/customer_churn_historical.csv
git add data/raw/customer_churn_historical.csv.dvc data/raw/.gitignore .dvc/config
git commit -m "Track historical churn dataset with DVC"
dvc push
```

Las credenciales de DagsHub deben configurarse fuera del repositorio (`.dvc/config.local`, credential helper o variables de entorno).

## 8. MLflow y Model Registry

### Opción A — MLflow local reproducible

```bash
python -m src.training.train \
  --data data/raw/customer_churn_historical.csv \
  --experiment customer-churn-entrega-1 \
  --register-best
```

Por defecto utiliza `sqlite:///mlflow.db`, que permite conservar los Runs y el Registry localmente.

UI:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

### Opción B — MLflow remoto en DagsHub

Configurar `MLFLOW_TRACKING_URI` y autenticación según el repositorio DagsHub del equipo y ejecutar el mismo comando. No se incluyen tokens ni claves en este proyecto.

## 9. Reproducción end-to-end de la Entrega 1

```bash
git clone <URL_REPOSITORIO_GITHUB>
cd <REPOSITORIO>
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
dvc pull
python scripts/eda.py
python -m src.training.train --register-best
```

Para la demostración del parcial debe poder mostrarse:

1. `git log` y tag `entrega-1`.
2. `dvc status` / `dvc pull` y remote DagsHub funcional.
3. Seis Runs comparables en MLflow.
4. Parámetros, métricas y artefactos de cada Run.
5. Modelo candidato en Model Registry y su `run_id` de origen.
6. Ejecución de training desde Python sin abrir el notebook.

## 10. Git y tag de entrega

Sobre el commit exacto presentado:

```bash
git add .
git commit -m "Entrega 1: proyecto reproducible Git DVC MLflow Registry"
git tag entrega-1
git push origin main
git push origin entrega-1
```

## 11. Estructura

```text
.
├── data/
│   ├── raw/
│   ├── production/
│   └── scoring/
├── metadata/
├── examples/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── training/
│   ├── evaluation/
│   └── inference/
├── models/
├── results/
├── scripts/
├── .dvc/
├── requirements.txt
└── README.md
```

## 12. Pendientes que dependen de las cuentas del equipo

Este paquete deja preparada la entrega, pero **no puede completar por sí solo** tres evidencias externas porque requieren credenciales/URLs del equipo:

- repositorio GitHub remoto y acceso para el profesor;
- remote DVC real en DagsHub y `dvc push` exitoso;
- proyecto/Tracking URI de MLflow en DagsHub, si se exige evidencia remota.

Una vez configuradas esas cuentas, ejecutar los comandos anteriores, verificar las evidencias y crear/pushear el tag `entrega-1`.
