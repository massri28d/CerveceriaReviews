import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline


# Crear categorías a partir de la calificación general
def categorize_review(data: pd.DataFrame) -> pd.DataFrame:
    print("Categorizando calificaciones generales...")
    bins = [0, 2, 4, 5]
    labels = ['Baja', 'Media', 'Alta']
    data['review_overall_cat'] = pd.cut(data['review_overall'], bins=bins, labels=labels, include_lowest=True)
    data.dropna(subset=['review_overall_cat'], inplace=True)
    print("Categorización completada.")
    return data

# Definir características y variable objetivo
def split_features_target(data: pd.DataFrame):
    print("Dividiendo características y variable objetivo...")
    X = data.drop(['review_overall_cat', 'review_overall', 'review_profilename', 'beer_name', 'beer_style', 'brewery_name'], axis=1)
    y = data['review_overall_cat']
    X = pd.get_dummies(X, drop_first=True)  # One-hot encoding
    print("División completada.")
    return X, y

# Dividir los datos en entrenamiento y prueba
def split_data(X: pd.DataFrame, y: pd.Series):
    print("Dividiendo datos en entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print("División completa.")
    return X_train, X_test, y_train, y_test

# Entrenar el modelo
def train_decision_tree(X_train, y_train):
    print("Entrenando modelo...")
    clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=2, random_state=42)
    clf.fit(X_train, y_train)
    print("Modelo entrenado.")
    return clf

# Evaluar el modelo
def evaluate_model(clf, X_test, y_test):
    print("Evaluando modelo...")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"Accurateidad: {accuracy:.2f}")
    print("Informe de clasificación:")
    print(report)
    print("Modelo evaluado.")
    return accuracy, report

# Visualizar el árbol de decisión
def visualize_tree(clf, feature_names):
    print("Visualizando árbol de decisión...")
    plt.figure(figsize=(10, 7))
    plot_tree(clf, filled=True, feature_names=feature_names, class_names=['Baja', 'Media', 'Alta'], rounded=True)
    plt.show()
    print("Árbol de decisión visualizado.")