from sklearn.metrics import accuracy_score, classification_report
#! Permite la medicion del modelo 
def evaluate_model (model, X_test,y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return{
        "accuracy": round(accuracy, 4),
        "report": report
       # "accuracy": accuracy_score(y_test, y_pred),
        #"report": classification_report(y_pred)
    } 