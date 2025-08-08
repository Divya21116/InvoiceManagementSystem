# from django.contrib import admin
# from .models import Company, Customer, Invoice, InvoiceItem

# class InvoiceItemInline(admin.TabularInline):
#     model = InvoiceItem
#     extra = 1

# @admin.register(Company)
# class CompanyAdmin(admin.ModelAdmin):
#     list_display = ['name', 'gstin', 'city', 'created_at']
#     search_fields = ['name', 'gstin']

# @admin.register(Customer)
# class CustomerAdmin(admin.ModelAdmin):
#     list_display = ['name', 'gstin', 'city', 'created_at']
#     search_fields = ['name', 'gstin']

# @admin.register(Invoice)
# class InvoiceAdmin(admin.ModelAdmin):
#     list_display = ['invoice_number', 'customer', 'document_type', 'grand_total', 'created_at']
#     list_filter = ['document_type', 'is_paid', 'created_at']
#     search_fields = ['invoice_number', 'customer__name']
#     inlines = [InvoiceItemInline]
#     readonly_fields = ['id', 'created_at', 'updated_at']
