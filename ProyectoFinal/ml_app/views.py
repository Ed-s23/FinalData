from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from .forms import DatasetUploadForms, PredictionForm
from .ml.data_loader import load_dataset
from .ml.pepeline import create_pipeline
from .ml.train import train_model
from .ml.evaluate import evaluate_model
from .ml.visualization import plot_protocol_type, plot_correlation_matrix


#! Funcion para el entrenamiento del modelo

def run_model(request):

    #! Apartado del GET 
    if request.method == 'GET':
        return render(request, 'upload.html', {
            'form': DatasetUploadForms()
        })

    #! Apartado de POST 
    if request.method == 'POST' and request.FILES.get('dataset'):
        form = DatasetUploadForms(request.POST, request.FILES)

        if not form.is_valid():
            return render(request, 'upload.html', {'form': form})

        #! Guardar archivo
        dataset_file = request.FILES['dataset']
        fs = FileSystemStorage()
        filename = fs.save(dataset_file.name, dataset_file)
        file_path = fs.path(filename)

        #!  Cargar dataset
        df = load_dataset(file_path)

        #! Target automático
        target = df.columns[-1]
        X = df.drop(columns=[target])
        y = df[target]

        #! Procsamiento de los tipos de columnas
        num_cols = X.select_dtypes(include='number').columns.tolist()
        cat_cols = X.select_dtypes(exclude='number').columns.tolist()

        #!Pipeline prosesado con el entrenamiento
        pipeline = create_pipeline(num_cols, cat_cols)
        model, X_test, y_test, dataset_info = train_model(X, y, pipeline)

        #!Guardar el modelo entrenado
        model_path = os.path.join(settings.MEDIA_ROOT, 'model.pkl')
        joblib.dump(model, model_path)

        #! Evaluación de los datos
        results = evaluate_model(model, X_test, y_test)
        results.update(dataset_info)

        #!  Visualizacion de la informacion
        results['plot_url'] = plot_protocol_type(df)
        #results['corr_img'] = plot_correlation_matrix(df)
        results['corr_img'] = plot_correlation_matrix(df)

        if 'class' in df.columns:
            le = LabelEncoder()
            df['class'] = le.fit_transform(df['class'])

        # Preview del dataset (primeras 20 filas)
        table_html = df.head(20).to_html(
            classes='data-table',
            index=False,
            border=0
        )

        results['table'] = table_html
        return render(request, 'results.html', results)



# PREDICCIÓN DESDE FORMULARIO

#def predict_view(request):
#    prediction = None

#    if request.method == 'POST':
#        form = PredictionForm(request.POST)
#
#        if form.is_valid():
#            data = form.cleaned_data
#            df_input = pd.DataFrame([data])

#            model_path = os.path.join(settings.MEDIA_ROOT, 'model.pkl')
#            model = joblib.load(model_path)

#            prediction = model.predict(df_input)[0]

#    else:
#        form = PredictionForm()

#    return render(request, 'predict.html', {
#        'form': form,
#        'prediction': prediction
#    })
