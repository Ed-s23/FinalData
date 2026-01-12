from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer


def create_pipeline(num_cols, cat_cols):

    transformers = []

    # 🔢 Pipeline para columnas numéricas
    if num_cols:
        numeric_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(
            ('num', numeric_pipeline, num_cols)
        )

    # 🔤 Pipeline para columnas categóricas
    if cat_cols:
        categorical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=False
            ))
        ])
        transformers.append(
            ('cat', categorical_pipeline, cat_cols)
        )

    # 🧠 Preprocesador completo
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )

    # 🤖 Pipeline final (preprocesamiento + modelo)
    model = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('classifier', LogisticRegression(
            max_iter=2000,
            n_jobs=-1,
            solver='lbfgs'
        ))
    ])

    return model
