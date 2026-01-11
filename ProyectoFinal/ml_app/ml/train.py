from sklearn.model_selection import train_test_split
#! Entrena el modelo con datos reales
def train_model ( df, pipeline, target):
    X = df.drop(target, axis= 1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    pipeline.fit(X_train,y_train)
    return pipeline, X_test, y_test