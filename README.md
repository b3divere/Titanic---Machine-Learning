# Titanic - Machine Learning

Proyecto introductorio de ciencia de datos en Python: análisis exploratorio, limpieza de datos y modelos de clasificación para predecir la supervivencia de los pasajeros del Titanic.

## Pregunta del proyecto

¿Qué factores influyeron en la supervivencia de un pasajero y se puede predecir quién sobrevivió?

## Datos

Dataset de la competencia [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic) de Kaggle. Se usa `train.csv`: 891 pasajeros, con la columna objetivo `Survived` (0 = murió, 1 = sobrevivió).

## Proceso

1. **Exploración:** revisión de tipos, nulos y estadísticas (`head`, `info`, `describe`, `isnull`).
2. **Limpieza:**
   - `Cabin` se elimina (77% de valores nulos).
   - `Age` se rellena con la mediana.
   - `Embarked` se rellena con la moda.
3. **Variables nuevas:**
   - `Familia` = `SibSp` + `Parch` + 1
   - `Solo` = 1 si el pasajero viajaba sin familia
   - `RangoEdad` (con `pd.cut`), usada solo en la exploración
4. **Preparación para el modelo:** se eliminan `PassengerId`, `Name`, `Ticket` y `RangoEdad`, y se codifican `Sex` y `Embarked` con `get_dummies`.
5. **Modelado:** división 80/20 en entrenamiento y prueba, regresión logística y Random Forest, más validación cruzada de 5 particiones.

## Hallazgos de la exploración

- Sobrevivió cerca del **74% de las mujeres** frente al **19% de los hombres**.
- Por clase: **63%** en primera, **47%** en segunda y **24%** en tercera.
- Combinando ambas: casi todas las mujeres de primera y segunda clase sobrevivieron (97% y 92%), mientras que los hombres de segunda y tercera clase rondan el 14-16%.
- Línea base: un modelo que prediga "todos murieron" acierta aproximadamente el 62%.

## Resultados

| Modelo | Accuracy (test 20%) | Validación cruzada (5 particiones) |
|---|---|---|
| Regresión logística | 79.9% | 79.2% |
| Random Forest (100 árboles) | 81.6% | 81.3% |

Importancia de variables (Random Forest): `Sex_male` (0.27), `Fare` (0.26) y `Age` (0.24) son las más relevantes; luego `Pclass` (0.08). Las variables de familia aportan poco una vez que se incluyen las demás.

## Conclusiones

- El **sexo** es el factor dominante de supervivencia en ambos modelos, seguido por la **clase** y la **tarifa**, que están relacionadas entre sí.
- El Random Forest rinde unos 2 puntos mejor, probablemente porque capta relaciones no lineales. La diferencia es pequeña y conviene interpretarla con cautela en un dataset de este tamaño.

## Estructura del proyecto

```
├── Data.py      # carga, limpieza y preparación de los datos
├── Model.py     # entrenamiento y evaluación de los modelos
├── train.csv    # datos de entrenamiento (Kaggle)
├── test.csv
└── gender_submission.csv
```

## Cómo ejecutarlo

```bash
pip install pandas scikit-learn seaborn matplotlib
python Model.py
```