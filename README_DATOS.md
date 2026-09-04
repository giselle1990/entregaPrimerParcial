# Datos de la Entrega 1

El proyecto utiliza `data/raw/customer_churn_historical.csv`, un dataset sintético de telecomunicaciones con 7.043 observaciones y la variable objetivo `Churn`.

Archivos asociados:

- `data/raw/customer_churn_historical.csv.dvc`: referencia utilizada por DVC;
- `metadata/data_dictionary.csv`: descripción y dominio de las columnas;
- `metadata/schema.json`: esquema del dataset histórico;
- `CHECKSUMS.sha256`: hash de los archivos de datos y metadatos incluidos.

Reglas aplicadas:

1. `customerID` se conserva para identificar registros, pero no se usa como predictor.
2. Los faltantes de `TotalCharges` se resuelven dentro del pipeline.
3. El CSV histórico se versiona con DVC y no se incorpora directamente a Git.
4. La división de entrenamiento, validación y test se genera de manera reproducible.
5. El dataset es sintético y no contiene datos personales reales.
