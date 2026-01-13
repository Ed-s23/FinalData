from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

from .forms import DatasetUploadForms
from .ml.data_loader import load_dataset
from .ml.pepeline import create_pipeline
from .ml.train import train_model
from .ml.evaluate import evaluate_model
from .ml.visualization import (
    plot_protocol_type,
    plot_correlation_matrix
)


def run_model(request):
    # ======================
    # GET → formulario
    # ======================
    if request.method == 'GET':
        return render(request, 'upload.html', {
            'form': DatasetUploadForms()
        })

    # ======================
    # POST → procesamiento
    # ======================
    if request.method == 'POST' and request.FILES.get('dataset'):
        form = DatasetUploadForms(request.POST, request.FILES)

        if not form.is_valid():
            return render(request, 'upload.html', {'form': form})

        # -------- Guardar dataset --------
        fs = FileSystemStorage()
        filename = fs.save(request.FILES['dataset'].name, request.FILES['dataset'])
        file_path = fs.path(filename)

        # -------- Cargar dataset --------
        df = load_dataset(file_path)

        # -------- Target automático --------
        target = df.columns[-1]
        X = df.drop(columns=[target])
        y = df[target]

        num_cols = X.select_dtypes(include='number').columns.tolist()
        cat_cols = X.select_dtypes(exclude='number').columns.tolist()

        # -------- Entrenamiento --------
        pipeline = create_pipeline(num_cols, cat_cols)
        model, X_test, y_test, dataset_info = train_model(X, y, pipeline)

        # -------- Guardar modelo --------
        model_path = os.path.join(settings.MEDIA_ROOT, 'model.pkl')
        joblib.dump(model, model_path)

        # -------- Evaluación --------
        results = evaluate_model(model, X_test, y_test)
        results.update(dataset_info)

        # ======================
        # CONTEOS (ANTES de encoding)
        # ======================

        # Conteo de clases
        if 'class' in df.columns:
            class_counts = df['class'].value_counts()
            results['class_data'] = class_counts.items()

            # Encoding solo para ML
            le = LabelEncoder()
            df['class_encoded'] = le.fit_transform(df['class'])

        # Conteo de protocolos (detecta nombre de columna)
        protocol_column = None
        for col in ['protocol_type', 'protocol', 'proto']:
            if col in df.columns:
                protocol_column = col
                break

        if protocol_column:
            protocol_counts = df[protocol_column].value_counts()
            results['protocol_data'] = protocol_counts.items()
        else:
            results['protocol_data'] = None

        # ======================
        # GRÁFICAS (STATIC)
        # ======================
        plot_protocol_type(df)
        plot_correlation_matrix(df)

        results['protocol_img'] = 'protocol_type.png'
        results['corr_img'] = 'matriz_correlacion.png'

        # ======================
        # PREVIEW DEL DATASET
        # ======================
        results['table'] = df.head(20).to_html(
            classes='data-table',
            index=False,
            border=0
        )

        return render(request, 'results.html', results)
