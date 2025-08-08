# import os
# import mimetypes
# from rest_framework.decorators import api_view
# from rest_framework.response import Response
# from rest_framework import status
# from .utils import detect_text, detect_fake, detect_image, detect_video

# # Define the directory to save uploaded files
# UPLOAD_DIR = r'C:\Users\divya.gugulothu\OneDrive - TECHWAVE\working\ISBHackthon\hackbackend\hackmain\uploaded'  # Update this path accordingly

# # Ensure the directory exists
# if not os.path.exists(UPLOAD_DIR):
#     os.makedirs(UPLOAD_DIR)

# @api_view(['POST'])
# def predict_file(request):
#     if request.method == 'POST':
#         # Check if text is provided
#         if 'text' in request.data:
#             input_text = request.data['text']
#             result = detect_text(input_text)  # Call the detect_text method
#             return Response({'result': result})

#         # Check if a file is provided
#         if 'file' not in request.FILES:
#             return Response({'error': 'No file provided'}, status=status.HTTP_400_BAD_REQUEST)

#         uploaded_file = request.FILES['file']
#         print("Uploaded file details:")
#         print(uploaded_file)

#         # Get the file type
#         file_type, _ = mimetypes.guess_type(uploaded_file.name)
#         print("Detected file type:")
#         print(file_type)

#         # Call the appropriate detection method based on the file type
#         if file_type is not None:
#             if file_type.startswith('audio/'):
#                 result = detect_fake(uploaded_file)
#             elif file_type.startswith('image/'):
#                 # Save the uploaded image to the specified directory
#                 image_file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
#                 with open(image_file_path, 'wb+') as destination:
#                     for chunk in uploaded_file.chunks():
#                         destination.write(chunk)

#                 # Call detect_image with the path of the saved image
#                 result = detect_image(image_file_path)
#             elif file_type.startswith('video/'):
#                 # Save the uploaded video to the specified directory
#                 video_file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
#                 with open(video_file_path, 'wb+') as destination:
#                     for chunk in uploaded_file.chunks():
#                         destination.write(chunk)

#                 # Call detect_video with the path of the saved video
#                 result = detect_video(video_file_path)
#             else:
#                 return Response({'error': 'Unsupported file type'}, status=status.HTTP_400_BAD_REQUEST)
#         else:
#             return Response({'error': 'Could not determine file type'}, status=status.HTTP_400_BAD_REQUEST)

#         return Response({'result': result})

#     return Response({'error': 'Invalid request method'}, status=status.HTTP_400_BAD_REQUEST)
# from rest_framework import viewsets, status
# from rest_framework.decorators import action
# from rest_framework.response import Response
# from django.http import HttpResponse
# from django.template.loader import get_template
# from .models import Company, Customer, Invoice, InvoiceItem
# from .serializers import CompanySerializer, CustomerSerializer, InvoiceSerializer, InvoiceCreateSerializer
# from .utils import generate_pdf
# import json

# class CompanyViewSet(viewsets.ModelViewSet):
#     queryset = Company.objects.all()
#     serializer_class = CompanySerializer

# class CustomerViewSet(viewsets.ModelViewSet):
#     queryset = Customer.objects.all()
#     serializer_class = CustomerSerializer

# class InvoiceViewSet(viewsets.ModelViewSet):
#     queryset = Invoice.objects.all()
    
#     def get_serializer_class(self):
#         if self.action == 'create':
#             return InvoiceCreateSerializer
#         return InvoiceSerializer

#     @action(detail=True, methods=['get'])
#     def generate_pdf(self, request, pk=None):
#         invoice = self.get_object()
#         pdf_content = generate_pdf(invoice)
        
#         response = HttpResponse(pdf_content, content_type='application/pdf')
#         response['Content-Disposition'] = f'attachment; filename="{invoice.invoice_number}.pdf"'
#         return response

#     @action(detail=False, methods=['get'])
#     def next_invoice_number(self, request):
#         document_type = request.query_params.get('type', 'invoice')
#         prefix = document_type[:3].upper()
        
#         last_invoice = Invoice.objects.filter(
#             document_type=document_type,
#             invoice_number__startswith=prefix
#         ).order_by('-created_at').first()
        
#         if last_invoice:
#             try:
#                 last_num = int(last_invoice.invoice_number.split('-')[-1])
#                 new_num = last_num + 1
#             except (ValueError, IndexError):
#                 new_num = 1
#         else:
#             new_num = 1
        
#         next_number = f"{prefix}-{new_num:03d}"
#         return Response({'next_number': next_number})
# invoice/views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import DocumentNumber1
import json

@csrf_exempt
def get_next_document_number(request):
    if request.method == "GET":
        print("hello")
        doc_type = request.GET.get("doc_type", "invoice")  # Default to invoice
        prefix_map = {"invoice": "INV", "challan": "CHA", "estimation": "EST"}
        prefix = prefix_map.get(doc_type, "INV")

        last_doc = DocumentNumber1.objects.filter(doc_type=doc_type).order_by("-id").first()
        next_number = (last_doc.number + 1) if last_doc else 1

        return JsonResponse({
            "document_number": f"{prefix}-{next_number:03d}"
        })


@csrf_exempt
def save_document_number(request):
    if request.method == "POST":
        data = json.loads(request.body)
        doc_type = data.get("doc_type", "invoice")
        document_number = data.get("document_number", "INV-001")

        prefix, number = document_number.split("-")
        number = int(number)

        DocumentNumber1.objects.create(
            doc_type=doc_type,
            prefix=prefix,
            number=number
        )

        return JsonResponse({"status": "success"})
