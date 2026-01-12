#from sklearn.pipeline import Pipeline
#from sklearn.linear_model import LogisticRegression
#from .pre_processing import get_preprocesso
#! Une preparación mas el modelo en un solo flujo 
#def create_pipeline(num_cols, cat_cols):
#    preprocessor = get_preprocesso(num_cols, cat_cols)

    #pipeline = Pipeline([
     #   ('preprocess', preprocessor)
      #  ('model',  LogisticRegression(max_iter=1000))
    #])

#    return pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer



"""def create_pipeline(num_cols, cat_cols):

    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    #preprocessor = ColumnTransformer(
    #    transformers=[
    #        (
    #            'num',StandardScaler(), num_cols
    #        ),
    #        ('cat', OneHotEncoder(handle_unknown='ignore'),cat_cols), 
    #    ]
    #)
    
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, num_cols),
        ('cat', categorical_pipeline, cat_cols)
    ])
    
    model = Pipeline(steps=[ 
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(max_iter=1000))
    ])

    return model"""
def create_pipeline(num_cols, cat_cols):

    transformers = []

    if len(num_cols) > 0:
        numeric_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        transformers.append(('num', numeric_pipeline, num_cols))

    if len(cat_cols) > 0:
        categorical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])
        transformers.append(('cat', categorical_pipeline, cat_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    return model


