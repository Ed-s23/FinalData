from django import forms

class DatasetUploadForms (forms.Form):
    dataset = forms.FileField(label= "subir los datos")