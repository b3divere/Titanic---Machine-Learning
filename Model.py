import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from Data import titulo, preparar_datos

# ---------- 1. Datos ----------
df_model = preparar_datos()

# ---------- 2. Separar ----------
X = df_model.drop(columns=["Survived"])
y = df_model["Survived"]

# ---------- 3. Dividir en entrenamiento y prueba(80% entrenamiento y 20% prueba) ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- 4. Regresion ----------
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
pred_log = log_reg.predict(X_test)
coeficientes = pd.Series(log_reg.coef_[0], index=X.columns)

# ---------- 5. Random Forest ----------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
importancias = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

# ---------- 6. Resultados ----------
titulo("Accuracy (conjunto de prueba)")
print("Logística: ", accuracy_score(y_test, pred_log))
print("Random Forest: ", accuracy_score(y_test, pred_rf))

titulo("Coeficientes de la regresión logística")
print(coeficientes)

titulo("variables (Random Forest)")
print(importancias)

# ---------- 7. Validacion cruzada ----------
titulo("Validacion cruzada (promedio de 5 pruebas)")
print("Logística: ", cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=5).mean())
print("Random Forest: ", cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42), X, y, cv=5).mean())

