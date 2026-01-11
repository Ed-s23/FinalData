import pandas as pd 
#! convierte el notebook de visualizacion en codigo reutilizable
def load_dataset(path): 
    df = pd.read_csv(path)
    return df