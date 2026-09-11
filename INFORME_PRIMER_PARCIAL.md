# Informe del primer parcial

## Objetivo

El trabajo busca detectar clientes con riesgo de churn y, al mismo tiempo, construir un entrenamiento que se pueda repetir y auditar. Por eso se versionan el código con Git, los datos con DVC y los experimentos con MLflow.

## Datos y preparación

El histórico tiene 7.043 clientes y 21 columnas. Hay 1.857 casos de churn (26,37%). `customerID` se descarta porque identifica al cliente pero no aporta una característica generalizable.

`TotalCharges` tiene 26 valores faltantes. No se rellenan directamente en el CSV: la imputación queda dentro del pipeline para aplicar la misma transformación en cada entrenamiento.

El preprocesamiento combina:

- mediana y escalado para las variables numéricas;
- moda y One-Hot Encoding para las categóricas;
- tolerancia a categorías no vistas.

## Separación de datos

La división es estratificada y usa `random_state=42`:

- 60% entrenamiento;
- 20% validación;
- 20% test.

Los seis modelos se comparan en validación. El test queda aislado hasta que el candidato ya fue elegido. Después, el modelo seleccionado se vuelve a ajustar con el 80% disponible y se evalúa una sola vez sobre test.

## Comparación y decisión

Se probaron un baseline, tres configuraciones de regresión logística y dos de Random Forest. Para la decisión se calculó `0,50*Recall + 0,30*F1 + 0,20*ROC-AUC`.

El mayor peso de Recall responde al costo del falso negativo: un cliente que realmente abandona queda fuera de una campaña de retención. No se usa Recall solo, porque un número excesivo de falsos positivos también tiene costo operativo.

El mejor resultado de validación fue `rf_300_depth12_t0.5`:

| Métrica de validación | Valor |
|---|---:|
| Precision | 0,523 |
| Recall | 0,701 |
| F1 | 0,599 |
| ROC-AUC | 0,804 |
| Falsos negativos | 111 |

Evaluación final sobre el test aislado:

| Métrica de test | Valor |
|---|---:|
| Accuracy | 0,747 |
| Precision | 0,516 |
| Recall | 0,634 |
| F1 | 0,569 |
| ROC-AUC | 0,801 |
| Falsos positivos | 221 |
| Falsos negativos | 136 |

La diferencia entre validación y test es esperable. Se informa sin volver a ajustar la selección sobre test, porque hacerlo contaminaría la evaluación final.

## Trazabilidad

MLflow recibe seis corridas de comparación y una corrida final. En cada una se guardan parámetros, métricas, pipeline, hash SHA-256 del dataset y commit de Git. La corrida final es la fuente del modelo `customer-churn-candidate` en Model Registry.

El pipeline completo también se serializa como `models/churn_pipeline.joblib`, de acuerdo con el flujo trabajado en clase. El módulo `src/inference/predict.py` carga ese artefacto y genera predicciones sin volver a entrenar.

El CSV histórico está ignorado por Git y referenciado por `data/raw/customer_churn_historical.csv.dvc`. El remote ya apunta a `https://dagshub.com/giselle.san/entregaPrimerParcial.dvc` y `dvc push` fue comprobado. Falta verificar la recuperación desde una segunda copia del repositorio.

La carpeta `app/` se conserva como lugar para una API futura, tal como indican las clases. Esta entrega mantiene una inferencia sencilla por consola y dos pruebas automáticas para métricas y preprocesamiento.

## Limitaciones

- La función de selección ponderada es una decisión académica; en un caso real debería definirse con costos de campaña y pérdida de clientes.
- No se hizo una búsqueda exhaustiva de hiperparámetros.
- Todavía no hay validación temporal: el dataset histórico se divide de forma aleatoria y estratificada.
- Las evidencias remotas de GitHub, DagsHub y MLflow dependen de las cuentas del equipo y deben completarse antes de entregar.
