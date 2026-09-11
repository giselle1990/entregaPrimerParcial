# Checklist de entrega

## Verificado en local

- [x] El entrenamiento se ejecuta desde Python sin abrir el notebook.
- [x] Se crean seis corridas de comparación y una corrida final.
- [x] Cada corrida guarda parámetros, métricas, tags y el pipeline.
- [x] El código registra `customer-churn-candidate` en Model Registry al usar `--register-best`.
- [x] El test queda aislado hasta la evaluación final.
- [x] El README incluye instalación y reproducción.
- [x] Las pruebas automáticas simples terminan correctamente.
- [x] El pipeline completo se guarda en `models/churn_pipeline.joblib`.
- [x] La inferencia se ejecuta desde `src/inference/predict.py`.

## Falta completar con las cuentas del equipo

- [x] URL del repositorio GitHub: `https://github.com/giselle1990/entregaPrimerParcial`
- [x] El repositorio es público y tiene acceso para el profesor.
- [x] El tag requerido `entrega-1` apunta al commit definitivo de esta revisión.
- [x] URL del proyecto DagsHub: `https://dagshub.com/giselle.san/entregaPrimerParcial`
- [x] `.dvc/config` contiene el remote real, no `USUARIO/REPOSITORIO`.
- [x] `dvc push` termina correctamente con `Everything is up to date`.
- [x] Otra copia del repositorio ejecutó `dvc pull` con credenciales locales, recuperó el CSV y pudo correr el EDA y los tests.
- [x] URL del experimento MLflow remoto: `https://dagshub.com/giselle.san/entregaPrimerParcial.mlflow`
- [x] Se ven las siete corridas completas en el servidor que se mostrará en la defensa.
- [x] Versión remota registrada: `3`.
- [x] Run ID remoto de origen: `a9013766e57641cc9fa53de5d72b92cd`.
- [ ] Otro integrante reprodujo el flujo completo.

La corrida final puede consultarse en:
`https://dagshub.com/giselle.san/entregaPrimerParcial.mlflow/#/experiments/0/runs/a9013766e57641cc9fa53de5d72b92cd`.

Además de las siete corridas completas quedó una corrida inicial adicional de `dummy_most_frequent`, porque Windows interrumpió el primer intento al imprimir un carácter de consola. Esa corrida no intervino en la selección final.

## Comprobaciones rápidas antes del cierre

```bash
git status
git log --oneline --decorate -5
git remote -v
git tag --points-at HEAD
dvc remote list
dvc status
```

No marcar una evidencia externa como completa hasta probarla desde una segunda copia del repositorio.
