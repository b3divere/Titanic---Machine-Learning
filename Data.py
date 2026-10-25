import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

def titulo(texto):
    print("\n" + "=" * 10, texto, "=" * 10)
    

def preparar_datos():
    df = pd.read_csv("train.csv")

    titulo("Primeras filas")
    print(df.head())

    titulo("Info")
    df.info()

    titulo("Estadisticas")
    print(df.describe())

    titulo("Nulos antes de limpiar")
    print(df.isnull().sum())

    # ---------- Edades ---------- #
    titulo("Edades")
    limites = [0, 12, 18, 40, 60, 100]
    etiquetas = ["Niño", "Joven", "Adulto", "Maduro", "Mayor"]

    df["RangoEdad"] = pd.cut(df["Age"], bins=limites, labels=etiquetas)

    print(df["RangoEdad"].value_counts(dropna=False))
    print(df.groupby("RangoEdad", observed=True)["Survived"].mean())

    # ---------- Limpieza de datos ---------- #
    df = df.drop(columns=['Cabin'])
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    titulo("Nulos despues de limpiar")
    print(df.isnull().sum())

    # ---------- Supervivencia ---------- #
    titulo("Supervivencia")
    print(df["Survived"].value_counts())
    print(df.groupby("Survived")["Pclass"].value_counts())
    print(df.groupby("Sex")["Survived"].mean())
    print(df.groupby("Pclass")["Survived"].mean())
    print(df.groupby(["Sex", "Pclass"])["Survived"].mean())
    print(df["Embarked"].value_counts())

    # ---------- Supervivencia Familia/Solo ---------- #
    df["Familia"] = df["Parch"] + df["SibSp"] + 1
    df["Solo"] = (df["Familia"] == 1).astype(int)

    titulo("Familias / Solo")
    print(df["Familia"].value_counts())
    print(df["Solo"].value_counts())
    print(df.groupby("Familia")["Survived"].mean())
    print(df.groupby("Solo")["Survived"].mean())
    print(df.groupby("Survived")["Fare"].mean())

    # ---------- Grafico ---------- #
    plt.figure()
    df["Age"].hist()
    plt.title("Distribucion por edad")

    plt.figure()
    sns.countplot(data=df, x="Pclass", hue="Survived")
    plt.title("Supervivencia por clase")

    df_model = df.drop(columns=["PassengerId", "Name", "Ticket", "RangoEdad"])
    df_model = pd.get_dummies(df_model, columns=["Embarked", "Sex"], drop_first=True, dtype=int)

    titulo("Dataframe para el modelo")
    print(df_model.head())
    df_model.info()

    plt.show()

    return df_model

