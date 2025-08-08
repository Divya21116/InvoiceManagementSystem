from rest_framework import serializers
from .models import Company, Customer, Invoice, InvoiceItem

class CompanySerializer(serializers.ModelSerializer):
    class Meta:
        model = Company
        fields = '__all__'

class CustomerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Customer
        fields = '__all__'

class InvoiceItemSerializer(serializers.ModelSerializer):
    class Meta:
        model = InvoiceItem
        fields = '__all__'

class InvoiceSerializer(serializers.ModelSerializer):
    items = InvoiceItemSerializer(many=True, read_only=True)
    customer_details = CustomerSerializer(source='customer', read_only=True)
    company_details = CompanySerializer(source='company', read_only=True)

    class Meta:
        model = Invoice
        fields = '__all__'

class InvoiceCreateSerializer(serializers.ModelSerializer):
    items = InvoiceItemSerializer(many=True)

    class Meta:
        model = Invoice
        fields = '__all__'

    def create(self, validated_data):
        items_data = validated_data.pop('items')
        invoice = Invoice.objects.create(**validated_data)
        
        total_amount = 0
        for item_data in items_data:
            item = InvoiceItem.objects.create(invoice=invoice, **item_data)
            total_amount += item.amount
        
        # Update invoice totals
        invoice.subtotal = total_amount + invoice.transportation_expenses
        invoice.tax_amount = (invoice.subtotal * invoice.cgst_sgst_rate) / 100
        invoice.grand_total = invoice.subtotal + invoice.tax_amount
        invoice.save()
        
        return invoice