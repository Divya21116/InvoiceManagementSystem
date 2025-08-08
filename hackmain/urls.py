# from django.urls import path
# from . import views

# urlpatterns = [
#     path('predict_file', views.predict_file, name='predict_file'),
# ]
from django.urls import path
from . import views


urlpatterns = [
    path('api/get-next-document-number', views.get_next_document_number),
    path('api/save-document-number/', views.save_document_number),
]