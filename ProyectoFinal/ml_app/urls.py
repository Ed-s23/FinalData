from django.urls import path 
from .views import run_model

urlpatterns = [
   # path('', home, name='home'),
   path ('', run_model,name = 'upload_dataset')
   # path('run/', run_model, name='run_model'),
]