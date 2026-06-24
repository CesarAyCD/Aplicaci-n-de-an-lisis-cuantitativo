import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42

TARGET_COL = "target_direction"
TARGET_RETURN_COL = "target_return"

# Elige todas las variables numéricas como explicativas. Evitando las que contienen información que el modelo no debe conocer.
def get_feature_columns(df):
    exclude_cols = {
        TARGET_COL,
        TARGET_RETURN_COL,
        "date",
        "session_date",
        "date_original",
        "triple_barrier_label",
        "barrier_touch_bars",
        "target_volatility",
        "close",
        "high",
        "low",
    }

    numeric_cols = [
        col
        for col, dtype in df.dtypes.items()
        if np.issubdtype(dtype, np.number) and df[col].notna().any()
    ]
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]

    return feature_cols

# Crea las matrices de entrenamiento
def make_xy(df, feature_cols):
    X = align_features(df, feature_cols)
    y = df[TARGET_COL].astype(np.int8, copy=False)
    return X, y

# Garantiza que la serie de tiempo esté ordenada
def align_features(df, feature_cols):
    return df.reindex(columns=feature_cols).astype(np.float32, copy=False)

# Inicializa los modelos usando pipelines de scikit-learn para evitar data leakage
def build_models():
    models = {}

    models["logistic_regression"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])

    models["random_forest"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=50,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    models["extra_trees"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesClassifier(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=50,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    models["hist_gradient_boosting"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.03,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        )),
    ])


    models["xgboost"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(
            n_estimators=400,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=20,
            reg_alpha=0.5,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    return models

# Ejecuta el entrenamiento
def fit_models(train, feature_cols):
    X_train, y_train = make_xy(train, feature_cols)

    models = build_models()
    fitted_models = {}

    for model_name, model in models.items():
        print(f"Entrenando modelo: {model_name}")
        model.fit(X_train, y_train)
        fitted_models[model_name] = model

    return fitted_models

# Extrae el espacio probabilístico en que el precio toca el take profit
def predict_model_probabilities(models, df, feature_cols):
    X = align_features(df, feature_cols)

    probabilities = {}

    for model_name, model in models.items():
        proba_up = model.predict_proba(X)[:, 1]
        probabilities[model_name] = proba_up

    return probabilities
