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

def create_pipeline(num_cols, cat_cols):

    num_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(max_iter=1000))
    ])

    return pipeline
