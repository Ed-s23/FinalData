from django.shortcuts import render
from .ml.data_loader import load_dataset
from .ml.pepeline import create_pipeline
from .ml.train import train_model
from .ml.evaluate import evaluate_model
#! Ejecucion de todos los proyectos dentro de ml
#def run_model(request):
#    df = load_dataset('DataSet/KDDTest+.txt') 

#    num_cols = df.select_dtypes(include='number').columns
#    cat_cols = df.select_dtypes(exclude='number').columns.drop('label')
#    pipeline = create_pipeline(num_cols, cat_cols)
#    model,X_test,y_test = train_model(df, pipeline, 'label')
#    results = evaluate_model(model,X_test,y_test)

 #   return render (request, 'results.html', results)

#def run_model(request):
#    df = load_dataset('DataSet/KDDTest+.txt')
#    print(df.columns)

#    target = 'label'

#    num_cols = df.select_dtypes(include='number').columns
#    cat_cols = [
#    col for col in df.select_dtypes(exclude='number').columns
#    if col != target
#]


    #pipeline = create_pipeline(num_cols, cat_cols)

    #model, X_test, y_test = train_model(df, pipeline, 'label')

#    results = evaluate_model(model, X_test, y_test)

 #   return render(request, 'results.html', results)
    

#def run_model(request):
 #        return render(request, 'results.html', {
 #       "accuracy": "OK",
 #       "report": "Funciona correctamente"
 #   })

# Create your views here.
def run_model(request):
    df = load_dataset('DataSet/KDDTest+.txt')

    target = 'label'

#    num_cols = df.select_dtypes(include='number').columns

#    cat_cols = [
#        col for col in df.select_dtypes(exclude='number').columns
#        if col != target
#    ]
    X = df.drop(columns=[target])
    y = df[target]
    num_cols = X.select_dtypes(include='number').columns.tolist()
    cat_cols = X.select_dtypes(exclude='number').columns.tolist()


    pipeline = create_pipeline(num_cols, cat_cols)

    model, X_test, y_test = train_model(df, pipeline, target)

    results = evaluate_model(model, X_test, y_test)

    return render(request, 'results.html', results)
