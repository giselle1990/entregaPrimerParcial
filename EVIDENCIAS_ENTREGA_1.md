# Checklist de entrega

## Verificado en local

- [x] El entrenamiento se ejecuta desde Python sin abrir el notebook.
- [x] Se crean seis corridas de comparación y una corrida final.
- [x] Cada corrida guarda parámetros, métricas, tags y el pipeline.
- [x] El código registra `customer-churn-candidate` en Model Registry al usar `--register-best`.
- [x] El test queda aislado hasta la evaluación final.
- [x] El README incluye instalación y reproducción.

## Falta completar con las cuentas del equipo

- [x] URL del repositorio GitHub: `https://github.com/giselle1990/entregaPrimerParcial`
- [x] El repositorio es público y tiene acceso para el profesor.
- [x] El tag `entrega-1` apunta al commit definitivo.
- [ ] URL del proyecto DagsHub: `____________________________`
- [ ] `.dvc/config` contiene el remote real, no `USUARIO/REPOSITORIO`.
- [ ] `dvc push` termina correctamente.
- [ ] Otra copia del repositorio puede ejecutar `dvc pull`.
- [ ] URL del experimento MLflow remoto: `____________________________`
- [ ] Se ven las siete corridas en el servidor que se mostrará en la defensa.
- [ ] Versión remota registrada: `________`.
- [ ] Run ID remoto de origen: `________________________________`.
- [ ] Otro integrante reprodujo el flujo completo.

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
