import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# ============================================================
# 1. Cargar el dataset
# ============================================================

def load_data(train_path, test_path):
    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race",
        "sex", "capital-gain", "capital-loss", "hours-per-week",
        "native-country", "income"
    ]

    train = pd.read_csv(train_path, names=columns, sep=",", skipinitialspace=True)
    test = pd.read_csv(test_path, names=columns, sep=",", skipinitialspace=True, skiprows=1)

    data = pd.concat([train, test], ignore_index=True)
    data = data.replace("?", pd.NA).dropna()
    return data

# ============================================================
# 2. Preprocesar los datos
# ============================================================

def preprocess_data(df):
    label_encoders = {}
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders

# ============================================================
# 3. Dividir en Train / Dev / Test
# ============================================================

def split_data(df):
    X = df.drop(columns=["income"])
    y = df["income"]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=2/3, random_state=42, stratify=y_temp)

    return X_train, X_dev, X_test, y_train, y_dev, y_test

# ============================================================
# 4. Entrenar modelos
# ============================================================

def train_models(X_train, y_train):
    models = {
        "Decision Tree": DecisionTreeClassifier(criterion="entropy", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
    return models

# ============================================================
# 5. Evaluar modelos
# ============================================================

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1": f1_score(y, y_pred)
    }

# ============================================================
# 6. Pipeline completo
# ============================================================

def main():
    data = load_data("../Parte 2/resources/adult.data", "../Parte 2/resources/adult.test")
    data, encoders = preprocess_data(data)
    X_train, X_dev, X_test, y_train, y_dev, y_test = split_data(data)

    models = train_models(X_train, y_train)

    print("=== Evaluación en DEV (Selección de modelo) ===")
    results_dev = {}
    for name, model in models.items():
        metrics = evaluate_model(model, X_dev, y_dev)
        results_dev[name] = metrics
        print(f"\n{name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    best_model_name = max(results_dev, key=lambda m: results_dev[m]["F1"])
    best_model = models[best_model_name]

    print(f"\n>>> Mejor modelo: {best_model_name}\n")
    print("=== Evaluación en TEST ===")
    test_metrics = evaluate_model(best_model, X_test, y_test)
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
