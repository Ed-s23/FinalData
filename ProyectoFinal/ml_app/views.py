from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import pandas as pd
from .forms import DatasetUploadForms
from .ml.data_loader import load_dataset
from .ml.pepeline import create_pipeline
from .ml.train import train_model
from .ml.evaluate import evaluate_model


def run_model(request):
    if request.method == 'POST' and request.FILES.get('dataset'):
        form = DatasetUploadForms(request.POST, request.FILES)

        if form.is_valid():
            dataset_file = request.FILES['dataset']

            # Guardar archivo en /media
            fs = FileSystemStorage()
            filename = fs.save(dataset_file.name, dataset_file)
            file_path = fs.path(filename)  # Ruta real en disco

            # Cargar dataset
            df = load_dataset(file_path)

            # Variable objetivo
            target = 'label'

            # Separar X e y
            X = df.drop(columns=[target])
            y = df[target]

            # Detectar columnas
            num_cols = X.select_dtypes(include='number').columns.tolist()
            cat_cols = X.select_dtypes(exclude='number').columns.tolist()
            # Forzar tipos correctos
            for col in cat_cols:
                X[col] = X[col].astype(str)

            for col in num_cols:
                X[col] = pd.to_numeric(X[col], errors='coerce')

            # 🔍 Debug (sale en la terminal)
            print("Columnas del DataFrame:", df.columns.tolist())
            print("Numéricas:", num_cols)
            print("Categóricas:", cat_cols)

            # Crear pipeline y entrenar
            pipeline = create_pipeline(num_cols, cat_cols)
            model, X_test, y_test = train_model(X, y, pipeline)

            # Evaluar modelo
            results = evaluate_model(model, X_test, y_test)

            return render(request, 'results.html', results)

    # GET (mostrar formulario)
    form = DatasetUploadForms()
    return render(request, 'upload.html', {'form': form})
