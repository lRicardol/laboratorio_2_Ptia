import pandas as pd
from main.Parte1.decision_tree_classifier import DecisionTreeClassifier

X = pd.DataFrame({
    'Clima': ['sol', 'sol', 'lluvia', 'sol', 'lluvia'],
    'Temperatura': ['alta', 'alta', 'baja', 'media', 'baja']
})
Y = pd.Series(['jugar', 'jugar', 'no_jugar', 'jugar', 'no_jugar'])

# Entrenar el modelo
model = DecisionTreeClassifier(criterion_name="entropy")
model.fit(X, Y)

# Mostrar el árbol
print("Árbol generado:")
model.print_tree()

# Predicciones
X_test = pd.DataFrame({
    'Clima': ['sol', 'lluvia'],
    'Temperatura': ['baja', 'alta']
})
Y_true = pd.Series(['jugar', 'no_jugar'])

Y_pred = model.predict(X_test)
print("\nPredicciones:")
print(Y_pred)

# Evaluar métricas
print("\nEvaluación:")
metrics = model.evaluate(X_test, Y_true)
for k, v in metrics.items():
    print(f"{k}: {v:.3f}")
