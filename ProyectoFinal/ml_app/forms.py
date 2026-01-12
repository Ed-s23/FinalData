from django import forms

class DatasetUploadForms(forms.Form):
    dataset = forms.FileField()

class PredictionForm(forms.Form):
    duration = forms.FloatField()
    src_bytes = forms.FloatField()
    dst_bytes = forms.FloatField()
    protocol_type = forms.CharField()
    service = forms.CharField()
    flag = forms.CharField()
