import pandas as pd 
from .columns import KDD_COLUMNS

#def load_dataset(path): 
#    return pd.read_csv(path,
#                       header=None,
#                       names=KDD_COLUMNS
#                       )
import pandas as pd
import os
import arff  
#! convierte el notebook de visualizacion en codigo reutilizable, archivos arff y csv
def load_dataset(path):
    ext = os.path.splitext(path)[1].lower()

    if ext == '.csv':
        df = pd.read_csv(path)

    elif ext == '.arff':
        with open(path, 'r') as f:
            arff_data = arff.load(f)

        df = pd.DataFrame(arff_data['data'])
        df.columns = [attr[0] for attr in arff_data['attributes']]

    else:
        raise ValueError("Formato no soportado (solo CSV y ARFF)")

    return df

    