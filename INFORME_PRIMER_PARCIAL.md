# Informe técnico — Primer Parcial
## Laboratorio de Minería de Datos — Customer Churn

## Objetivo de la etapa

Construir una primera versión reproducible y trazable del proyecto de Customer Churn, pasando del dataset/notebook exploratorio a un flujo Python versionable con Git, datos gestionados por DVC y experimentación/modelado preparada para MLflow y Model Registry.

## Datos utilizados

El histórico contiene 7.043 observaciones y 21 columnas. `Churn` es la variable objetivo y presenta 1.857 casos positivos (26,37%) y 5.186 negativos (73,63%). `customerID` se excluye del entrenamiento por ser un identificador. Se detectan 26 valores faltantes en `TotalCharges`, que se resuelven dentro del pipeline mediante imputación por mediana.

El archivo `customer_churn_current.csv` no se utiliza en esta etapa porque queda reservado para el análisis posterior de drift. El archivo `scoring_batch.csv` tampoco interviene en el entrenamiento inicial.

## Reproducibilidad

La partición de datos se realiza con `train_test_split`, test de 20%, `random_state=42` y estratificación por target. El preprocesamiento se integra en un `ColumnTransformer`: las variables numéricas se imputan y escalan; las categóricas se imputan con la moda y se codifican mediante One-Hot Encoding con tolerancia a categorías desconocidas.

Toda la lógica de entrenamiento se ejecuta desde `src/training/train.py`; el notebook queda limitado al EDA.

## Experimentación

Se comparan seis configuraciones relevantes: un baseline `DummyClassifier`, tres variantes de regresión logística —incluyendo un threshold de 0,35— y dos Random Forest. Se registran Accuracy, Precision, Recall, F1, ROC-AUC y matriz de confusión.

La mejor alternativa para el objetivo de negocio es `logreg_c1_t0.35`: Recall 0,6425, F1 0,5931, ROC-AUC 0,8120 y 133 falsos negativos. Con threshold 0,50 la misma regresión logística obtiene Recall 0,4516 y 204 falsos negativos. La reducción de falsos negativos justifica el threshold menor, aceptando más falsos positivos.

La selección se formaliza mediante `0.50*Recall + 0.30*F1 + 0.20*ROC-AUC`, priorizando Recall porque un falso negativo representa un cliente que realmente abandonará pero no será detectado como riesgo.

## Trazabilidad con MLflow

`src/training/train.py` registra para cada Run parámetros, métricas, tags y el pipeline entrenado. El Run seleccionado se vincula con el modelo `customer-churn-candidate` en Model Registry mediante `--register-best`. Además, se guarda el `run_id` del candidato en `results/selected_model.json` al ejecutar el entrenamiento con MLflow.

## Versionado de datos con DVC

`customer_churn_historical.csv` está excluido de Git y acompañado por `data/raw/customer_churn_historical.csv.dvc`. El proyecto deja preparado un remote `dagshub` y un script de configuración. La URL real del repositorio y las credenciales deben completarse con las cuentas del equipo; no se almacenan secretos en Git.

## Evidencias para la defensa

Antes de presentar, deben verificarse: tag Git `entrega-1`, remote DVC/DagsHub funcional, `dvc pull` reproducible, al menos seis Runs visibles en MLflow, modelo registrado y relación explícita con el Run de origen. El archivo `EVIDENCIAS_ENTREGA_1.md` funciona como checklist final.
