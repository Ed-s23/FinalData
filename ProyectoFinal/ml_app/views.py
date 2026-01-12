from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import pandas as pd
import matplotlib.pyplot as plt
import os

from .forms import DatasetUploadForms
from .ml.data_loader import load_dataset
from .ml.pepeline import create_pipeline
from .ml.train import train_model
from .ml.evaluate import evaluate_model
import matplotlib
matplotlib.use('Agg')  # MUY IMPORTANTE para Django
import matplotlib.pyplot as plt

def run_model(request):
    if request.method == 'POST' and request.FILES.get('dataset'):
        form = DatasetUploadForms(request.POST, request.FILES)

        if form.is_valid():
            dataset_file = request.FILES['dataset']

            # 📁 Guardar archivo
            fs = FileSystemStorage()
            filename = fs.save(dataset_file.name, dataset_file)
            file_path = fs.path(filename)

            # 📊 Cargar dataset
            df = load_dataset(file_path)

            # 🎯 Detectar columna objetivo
            target = df.columns[-1]
            X = df.drop(columns=[target])
            y = df[target]

            # 🔎 Detectar tipos
            num_cols = X.select_dtypes(include='number').columns.tolist()
            cat_cols = X.select_dtypes(exclude='number').columns.tolist()

            # Forzar tipos
            for col in cat_cols:
                X[col] = X[col].astype(str)

            for col in num_cols:
                X[col] = pd.to_numeric(X[col], errors='coerce')

            # 🧪 Debug
            print("Columnas:", df.columns.tolist())
            print("Numéricas:", num_cols)
            print("Categóricas:", cat_cols)

            # 🤖 Entrenamiento
            pipeline = create_pipeline(num_cols, cat_cols)
            model, X_test, y_test = train_model(X, y, pipeline)

            # 📈 Evaluación
            results = evaluate_model(model, X_test, y_test)

            # 📊 ----------- GRÁFICA protocol_type -----------
            plot_url = None
            corr_img = generate_correlation_plot(df)
            results['corr_img'] = corr_img

            if 'protocol_type' in df.columns:
                protocol_counts = df['protocol_type'].value_counts()

                plt.figure(figsize=(6, 4))
                plt.bar(protocol_counts.index, protocol_counts.values)
                plt.title('Distribución de protocol_type')
                plt.xlabel('Protocolo')
                plt.ylabel('Frecuencia')

                plot_path = os.path.join(settings.MEDIA_ROOT, 'protocol_plot.png')
                plt.savefig(plot_path)
                plt.close()

                plot_url = settings.MEDIA_URL + 'protocol_plot.png'

            # 📤 Enviar resultados + gráfica
            results['plot_url'] = plot_url

            return render(request, 'results.html', results)

    # GET
    form = DatasetUploadForms()
    return render(request, 'upload.html', {'form': form})
#! Generacion de grafica de correlacion
def generate_correlation_plot(df):
    numeric_df = df.select_dtypes(include='number')
    corr = numeric_df.corr()

    plt.figure(figsize=(12, 10))
    plt.imshow(corr)
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=6)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=6)
    plt.tight_layout()

    img_path = 'media/correlation_matrix.png'
    plt.savefig(img_path)
    plt.close()

    return img_path