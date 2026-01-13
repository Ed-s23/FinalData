from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose  import ColumnTransformer

#! Normaliza numeros y codifica texto 
def get_preprocesso (num_cols, cat_cols):
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'),cat_cols)
        ]
    )
