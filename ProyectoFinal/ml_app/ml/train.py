from sklearn.model_selection import train_test_split
import joblib
import os
from django.conf import settings
#! Entrena el modelo con datos reales
def train_model ( X, y, pipeline):
    #X = df.drop(target, axis= 1)
    #y = df[target]
    #*Permitimos el 80% entrenamoiento y el 20% testeo
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    #* permitimos el 10% val, y 10 % test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )  
    #* Entrenamiento 
    model = pipeline.fit(X_train,y_train)
    #* Acuraicy de validacion
    val_score = model.score(X_val, y_val)

    dataset_info = {
    "train_size": len(X_train),
    "val_size": len(X_val),
    "test_size": len(X_test),
    "total_size": len(X),
    "val_accuracy": round (val_score * 100,2)
    } 
    #!guarda el modelo para predecir datos del usuario
   # model_path = os.path.join(settings.MEDIA_ROOT, 'model.pkl')
    #joblib.dump(model, model_path)

   
    #pipeline.fit(X_train,y_train)
    return model, X_test, y_test, dataset_info 

#def train_model(X, y, pipeline):
#    from sklearn.model_selection import train_test_split

#    X_train, X_test, y_train, y_test = train_test_split(
#        X, y, test_size=0.2, random_state=42, stratify=y
    #)

#    model = pipeline.fit(X_train, y_train)

    # 💾 Guardar modelo
 #   model_path = os.path.join(settings.MEDIA_ROOT, 'model.pkl')
#    joblib.dump(model, model_path)

 #   return model, X_test, y_test
    
