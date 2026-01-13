from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
import joblib
import pandas as pd
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

    # =========================
    # FORMULARIO
    # =========================
    if request.method == 'GET':
        return render(request, 'upload.html', {
            'form': DatasetUploadForms()
        })

    if request.method == 'POST' and request.FILES.get('dataset'):
        form = DatasetUploadForms(request.POST, request.FILES)

        if not form.is_valid():
            return render(request, 'upload.html', {'form': form})

        # =========================
        # GUARDAR DATASET
        # =========================
        fs = FileSystemStorage()
        filename = fs.save(request.FILES['dataset'].name, request.FILES['dataset'])
        file_path = fs.path(filename)

        # =========================
        # CARGAR DATASET
        # =========================
        df = load_dataset(file_path)

        target = df.columns[-1]
        X = df.drop(columns=[target])
        y = df[target]

        num_cols = X.select_dtypes(include='number').columns.tolist()
        cat_cols = X.select_dtypes(exclude='number').columns.tolist()

        # =========================
        # ENTRENAMIENTO
        # =========================
        pipeline = create_pipeline(num_cols, cat_cols)
        model, X_test, y_test, dataset_info = train_model(X, y, pipeline)

        # =========================
        # GUARDAR MODELO
        # =========================
        model_path = os.path.join(settings.MEDIA_ROOT, 'model.pkl')
        joblib.dump(model, model_path)

        # =========================
        # EVALUACIÓN
        # =========================
        eval_results = evaluate_model(model, X_test, y_test)

        clean_report = {
            k: v for k, v in eval_results["report"].items()
            if isinstance(v, dict) and "precision" in v
        }

        # =========================
        # CONTEO DE CLASES
        # =========================
        class_data = None
        if 'class' in df.columns:
            class_data = df['class'].value_counts().items()

            le = LabelEncoder()
            df['class_encoded'] = le.fit_transform(df['class'])

        # =========================
        # CONTEO DE PROTOCOLOS
        # =========================
        protocol_data = None
        for col in ['protocol_type', 'protocol', 'proto']:
            if col in df.columns:
                protocol_data = df[col].value_counts().items()
                break

        # =========================
        # GRÁFICAS
        # =========================
        plot_protocol_type(df)
        plot_correlation_matrix(df)

        # =========================
        # PREVIEW DEL DATASET
        # =========================
        table_html = df.head(20).to_html(
            classes='data-table',
            index=False,
            border=0
        )

        # =========================
        # CONTEXTO FINAL
        # =========================
        context = {
            "accuracy": eval_results["accuracy"],
            "report": clean_report,
            "class_data": class_data,
            "protocol_data": protocol_data,
            "protocol_img": "protocol_type.png",
            "corr_img": "matriz_correlacion.png",
            "table": table_html,
        }
        clean_report = {}

        for label, metrics in eval_results["report"].items():
            if isinstance(metrics, dict) and "precision" in metrics:
                clean_report[label] = {
                    "precision": round(metrics["precision"], 4),
                    "recall": round(metrics["recall"], 4),
                    "f1_score": round(metrics["f1-score"], 4),
                    "support": metrics["support"],
                }

        context.update(dataset_info)

        return render(request, 'results.html', context)
