from django.urls import path 
from .views import run_model
from . import views 
urlpatterns = [
   # path('', home, name='home'),
   path ('', run_model,name = 'upload_dataset'),
  # path('predict/', views.predict_view, name='predict')

   # path('run/', run_model, name='run_model'),
]