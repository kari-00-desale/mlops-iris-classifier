import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def load_data():
    iris = load_iris()
    return train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )


def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def main():
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, predictions))

    joblib.dump(model, "models/iris_model.joblib")
    print("Model saved successfully.")


if __name__ == "__main__":
    main()