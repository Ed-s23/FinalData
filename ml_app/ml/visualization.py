import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from django.conf import settings

#! Generacion de Imagen Distribucion de Protocolo
def plot_protocol_type(df):
    if 'protocol_type' not in df.columns:
        return None

    counts = df['protocol_type'].value_counts()

    plt.figure(figsize=(6, 4))
    plt.bar(counts.index, counts.values)
    plt.title('Distribución de protocol_type')
    plt.xlabel('Protocolo')
    plt.ylabel('Frecuencia')
    plt.tight_layout()

    path = os.path.join(settings.MEDIA_ROOT, 'protocol_plot.png')
    plt.savefig(path, dpi=150)
    plt.close()

    return settings.MEDIA_URL + 'protocol_plot.png'

#! Generacion de imagen, Matriz de Correlacion
def plot_correlation_matrix(df):
    numeric_df = df.select_dtypes(include='number')

    if numeric_df.shape[1] < 2:
        return None

    numeric_df = numeric_df.iloc[:, :20]
    corr = numeric_df.corr().fillna(0)

    plt.figure(figsize=(14, 12))
    plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(fraction=0.046, pad=0.04)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=8)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=8)

    plt.title('Matriz de Correlación (Top 20 variables)')
    plt.tight_layout()

    path = os.path.join(settings.MEDIA_ROOT, 'correlation_matrix.png')
    plt.savefig(path, dpi=200)
    plt.close()

    return settings.MEDIA_URL + 'correlation_matrix.png'
import matplotlib.pyplot as plt
import os
from django.conf import settings

def plot_protocol_type(df):
    plt.figure()
    df['protocol_type'].value_counts().plot(kind='bar')
    plt.title("Distribución del protocolo")

    path = os.path.join(settings.STATIC_ROOT, "protocol_type.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()

def plot_correlation_matrix(df):
    plt.figure(figsize=(8,6))
    corr = df.select_dtypes(include='number').corr()
    plt.imshow(corr, cmap='coolwarm')
    plt.colorbar()
    plt.title("Matriz de correlación")

    path = os.path.join(settings.STATIC_ROOT, "matriz_correlacion.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
