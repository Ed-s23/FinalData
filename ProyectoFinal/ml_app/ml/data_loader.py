import pandas as pd 
from .columns import KDD_COLUMNS
#! convierte el notebook de visualizacion en codigo reutilizable
def load_dataset(path): 
    return pd.read_csv(path,
                       header=None,
                       names=KDD_COLUMNS
                       )
    
    