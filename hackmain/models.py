

# # Create your models here.
# from django.db import models
# from django.contrib.auth.models import User
# import uuid

# class Company(models.Model):
#     name = models.CharField(max_length=200)
#     subtitle = models.CharField(max_length=200, blank=True)
#     address = models.TextField()
#     city = models.CharField(max_length=100)
#     state = models.CharField(max_length=100)
#     pincode = models.CharField(max_length=10)
#     phone = models.CharField(max_length=100)
#     email = models.EmailField(blank=True)
#     gstin = models.CharField(max_length=15)
#     bank_name = models.CharField(max_length=100)
#     account_number = models.CharField(max_length=50)
#     branch = models.CharField(max_length=100)
#     ifsc = models.CharField(max_length=11)
#     authorized_signatory = models.CharField(max_length=100)
#     created_at = models.DateTimeField(auto_now_add=True)

#     def __str__(self):
#         return self.name

# class Customer(models.Model):
#     name = models.CharField(max_length=200)
#     address = models.TextField()
#     city = models.CharField(max_length=100, blank=True)
#     state = models.CharField(max_length=100, blank=True)
#     pincode = models.CharField(max_length=10, blank=True)
#     gstin = models.CharField(max_length=15, blank=True)
#     phone = models.CharField(max_length=100, blank=True)
#     email = models.EmailField(blank=True)
#     created_at = models.DateTimeField(auto_now_add=True)

#     def __str__(self):
#         return self.name

# class Invoice(models.Model):
#     DOCUMENT_TYPES = [
#         ('invoice', 'Invoice'),
#         ('challan', 'Delivery Challan'),
#         ('estimate', 'Estimate'),
#     ]
    
#     id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
#     document_type = models.CharField(max_length=20, choices=DOCUMENT_TYPES, default='invoice')
#     invoice_number = models.CharField(max_length=50, unique=True)
#     date = models.DateField()
#     company = models.ForeignKey(Company, on_delete=models.CASCADE)
#     customer = models.ForeignKey(Customer, on_delete=models.CASCADE)
    
#     # Additional fields
#     place_of_supply = models.CharField(max_length=100, blank=True)
#     vehicle_number = models.CharField(max_length=20, blank=True)
#     transport_mode = models.CharField(max_length=50, blank=True)
    
#     # Amounts
#     subtotal = models.DecimalField(max_digits=12, decimal_places=2, default=0)
#     transportation_expenses = models.DecimalField(max_digits=12, decimal_places=2, default=0)
#     cgst_sgst_rate = models.DecimalField(max_digits=5, decimal_places=2, default=18.00)
#     tax_amount = models.DecimalField(max_digits=12, decimal_places=2, default=0)
#     grand_total = models.DecimalField(max_digits=12, decimal_places=2, default=0)
    
#     # Status
#     is_paid = models.BooleanField(default=False)
#     notes = models.TextField(blank=True)
    
#     created_at = models.DateTimeField(auto_now_add=True)
#     updated_at = models.DateTimeField(auto_now=True)

#     class Meta:
#         ordering = ['-created_at']

#     def __str__(self):
#         return f"{self.invoice_number} - {self.customer.name}"

#     def save(self, *args, **kwargs):
#         if not self.invoice_number:
#             # Auto-generate invoice number
#             prefix = self.document_type[:3].upper()
#             last_invoice = Invoice.objects.filter(
#                 document_type=self.document_type,
#                 invoice_number__startswith=prefix
#             ).order_by('-created_at').first()
            
#             if last_invoice:
#                 try:
#                     last_num = int(last_invoice.invoice_number.split('-')[-1])
#                     new_num = last_num + 1
#                 except (ValueError, IndexError):
#                     new_num = 1
#             else:
#                 new_num = 1
            
#             self.invoice_number = f"{prefix}-{new_num:03d}"
        
#         super().save(*args, **kwargs)

# class InvoiceItem(models.Model):
#     invoice = models.ForeignKey(Invoice, related_name='items', on_delete=models.CASCADE)
#     description = models.TextField()
#     hsn_code = models.CharField(max_length=20, blank=True)
#     width = models.DecimalField(max_digits=10, decimal_places=2, default=0)
#     height = models.DecimalField(max_digits=10, decimal_places=2, default=0)
#     quantity = models.DecimalField(max_digits=10, decimal_places=2, default=1)
#     unit = models.CharField(max_length=20, default='Sft')
#     rate = models.DecimalField(max_digits=10, decimal_places=2)
#     amount = models.DecimalField(max_digits=12, decimal_places=2)

#     def save(self, *args, **kwargs):
#         # Calculate amount based on dimensions, quantity and rate
#         area = self.width * self.height
#         self.amount = area * self.quantity * self.rate
#         super().save(*args, **kwargs)

#     def __str__(self):
#         return f"{self.description} - {self.amount}"
# invoice/models.py
from django.db import models

class DocumentNumber1(models.Model):
    doc_type = models.CharField(max_length=20)
    number = models.IntegerField(default=1)   # this is the column SQLite says is missing
    prefix = models.CharField(max_length=10)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.prefix}-{self.number:03d}"

