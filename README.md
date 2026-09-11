# Customer Churn - Entrega 1

Proyecto del primer parcial de Laboratorio de Minería de Datos. El objetivo es predecir abandono de clientes de una empresa de telecomunicaciones y dejar trazabilidad entre datos, código, experimentos y modelo.

## Datos

El archivo de entrenamiento es `data/raw/customer_churn_historical.csv`: 7.043 filas, 21 columnas y una tasa de churn de 26,37%. `customerID` se elimina antes de entrenar porque es un identificador. Los 26 faltantes de `TotalCharges` se imputan dentro del pipeline.

Para regenerar el resumen exploratorio:

```bash
python scripts/eda.py
```

El resultado queda en `results/eda_summary.md`.

## Preparación y partición

El preprocesamiento se implementa con `Pipeline` y `ColumnTransformer` de scikit-learn:

- variables numéricas: imputación por mediana y estandarización;
- variables categóricas: imputación por moda y One-Hot Encoding;
- categorías nuevas: `handle_unknown="ignore"`.

Se usa una partición estratificada 60/20/20:

- 60% para ajustar las alternativas;
- 20% de validación para comparar modelos y elegir el threshold;
- 20% de test, que se consulta una sola vez después de elegir el candidato.

Todos los splits usan `random_state=42`.

## Experimentos

Se comparan seis configuraciones en validación:

| Configuración | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Dummy (baseline) | 0,000 | 0,000 | 0,000 | 0,500 |
| Logistic Regression, C=0,5 | 0,653 | 0,477 | 0,551 | 0,822 |
| Logistic Regression, C=1 | 0,657 | 0,480 | 0,555 | 0,821 |
| Logistic Regression, C=1, threshold=0,35 | 0,543 | 0,666 | 0,598 | 0,821 |
| Random Forest, 200 árboles | 0,553 | 0,620 | 0,584 | 0,797 |
| Random Forest, 300 árboles, profundidad 12 | 0,523 | 0,701 | 0,599 | 0,804 |

Para ordenar las alternativas se usa:

```text
0,50 * Recall + 0,30 * F1 + 0,20 * ROC-AUC
```

Recall tiene mayor peso porque un falso negativo es un cliente que va a abandonar y no sería incluido en una acción de retención. Precision y F1 siguen siendo relevantes: bajar falsos negativos a cualquier costo produciría demasiadas acciones comerciales innecesarias.

El candidato elegido en validación es `rf_300_depth12_t0.5`. Luego se reajusta con train + validación y se evalúa una sola vez en test:

| Precision | Recall | F1 | ROC-AUC | FP | FN |
|---:|---:|---:|---:|---:|---:|
| 0,516 | 0,634 | 0,569 | 0,801 | 221 | 136 |

La corrida final es una séptima corrida y es la que se registra como `customer-churn-candidate`. Así quedan separados el proceso de selección y la medición final.

## Instalación

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## DVC y DagsHub

El CSV histórico no está versionado por Git; lo referencia `data/raw/customer_churn_historical.csv.dvc`.

El remote DVC ya apunta al proyecto real de DagsHub:

```text
https://dagshub.com/giselle.san/entregaPrimerParcial.dvc
```

Las credenciales quedan en `.dvc/config.local`, que está ignorado por Git. Para comprobar el remote y sincronizar los datos:

```bash
python -m dvc remote list
python -m dvc push
dvc pull
dvc status
```

## Entrenamiento y MLflow

Ejecución local con registro del candidato:

```bash
python -m src.training.train --register-best
```

El backend local por defecto es `sqlite:///mlflow.db`. Para abrir la interfaz:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Para usar DagsHub se debe definir `MLFLOW_TRACKING_URI` y la autenticación correspondiente antes de ejecutar el mismo comando. Cada corrida guarda parámetros, métricas, tags de commit/dataset y el pipeline completo. La corrida final origina la versión registrada en Model Registry.

Los CSV y JSON generados por cada ejecución se guardan en `results/generated/`. Esa carpeta es local y se puede regenerar; la evidencia principal queda en MLflow.

## Modelo e inferencia

Además del registro en MLflow, el entrenamiento guarda el pipeline completo en `models/churn_pipeline.joblib`. El artefacto contiene el preprocesamiento y el modelo, como se trabajó en clase.

Para generar predicciones desde consola:

```bash
python -m src.inference.predict archivo_clientes.csv
```

El resultado se guarda en `results/predictions.csv`. La carpeta `app/` queda reservada para una futura API con FastAPI; no se agrega esa complejidad en esta entrega.

## Pruebas

```bash
pytest
```

Las pruebas iniciales comprueban el preprocesamiento ante faltantes y categorías nuevas, y el cálculo de métricas.

## Reproducción desde cero

```bash
git clone https://github.com/giselle1990/entregaPrimerParcial.git
cd entregaPrimerParcial
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
dvc pull
python scripts/eda.py
python -m src.training.train --register-best
pytest
```

En Linux o macOS, el comando de activación es `source .venv/bin/activate`.

## Estructura principal

```text
data/raw/              datos originales gestionados por DVC
data/processed/        salidas intermedias
data/reference/        datos de referencia
metadata/              esquema y diccionario de datos
notebooks/01_eda.ipynb exploración inicial
app/                   espacio reservado para la futura API
models/                artefacto joblib generado localmente
tests/                 pruebas automáticas simples
scripts/               comandos auxiliares
src/data/              carga y validación del dataset
src/features/          preprocesamiento
src/evaluation/        métricas
src/training/          entrenamiento y experimentación
src/inference/         carga del pipeline y predicción
results/               resultados reproducibles
README.md              instalación, ejecución y arquitectura
.env.example           variables necesarias sin secretos
```

## Antes de entregar

Todavía requieren evidencia externa:

- ejecutar las corridas en el Tracking Server que se mostrará en la defensa;
- completar `EVIDENCIAS_ENTREGA_1.md` con URLs, versión y run ID reales;
- comprobar la reproducción desde una segunda copia del repositorio.
