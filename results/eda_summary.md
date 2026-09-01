# EDA — Customer Churn histórico

- Dimensiones: **7043 filas × 21 columnas**.
- Tipos: **17 categóricas/objeto** y **4 numéricas**.
- Target `Churn`: No=5186 (73.63%), Yes=1857 (26.37%).
- Faltantes: TotalCharges=26.
- `customerID` se excluye del conjunto de features por ser identificador.
- `TotalCharges` se imputa dentro del pipeline para evitar tratamiento manual inconsistente.

## Distribución resumida de variables numéricas

|       |   SeniorCitizen |   tenure |   MonthlyCharges |   TotalCharges |
|:------|----------------:|---------:|-----------------:|---------------:|
| count |         7043    |  7043    |          7043    |        7017    |
| mean  |            0.17 |    35.17 |            68.17 |        2312.08 |
| std   |            0.38 |    18.9  |            24.98 |        1573.97 |
| min   |            0    |     0    |            18    |           0    |
| 25%   |            0    |    20    |            55.36 |         986.63 |
| 50%   |            0    |    35    |            73.91 |        2013.2  |
| 75%   |            0    |    51    |            88    |        3452.85 |
| max   |            1    |    72    |           114.41 |        7761.34 |

## Churn por tipo de contrato

| Contract       |    No |   Yes |
|:---------------|------:|------:|
| Month-to-month | 61.27 | 38.73 |
| One year       | 85.03 | 14.97 |
| Two year       | 91.03 |  8.97 |

## Churn por servicio de Internet

| InternetService   |    No |   Yes |
|:------------------|------:|------:|
| DSL               | 80.03 | 19.97 |
| Fiber optic       | 59.75 | 40.25 |
| No                | 92.38 |  7.62 |