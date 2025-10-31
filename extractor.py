import base64
import json
import re
from groq import Groq
from PIL import Image, ImageEnhance, ImageFilter

def extract_receipt_data(image_path, groq_api_key):
    client = Groq(api_key=groq_api_key)
    
    # Preprocess image for better OCR
    image = Image.open(image_path)
    
    # Convert to grayscale
    image = image.convert('L')
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)
    
    # Sharpen
    image = image.filter(ImageFilter.SHARPEN)
    
    # Save preprocessed image temporarily
    preprocessed_path = image_path.replace('.', '_processed.')
    image.save(preprocessed_path)
    
    # Read and encode preprocessed image
    with open(preprocessed_path, "rb") as img_file:
        image_data = base64.b64encode(img_file.read()).decode('utf-8')
    
    prompt = """Extract receipt data and return ONLY a JSON object with this exact structure (no markdown, no explanation):

{
  "header": {
    "merchant_name": "",
    "address": "",
    "tax_id": "",
    "contact": ""
  },
  "customer": {
    "name": "",
    "address": "",
    "tax_id": ""
  },
  "date": "",
  "line_items": [
    {"description": "", "amount": ""}
  ],
  "payment": {
    "total": "",
    "vat": "",
    "net_total": ""
  },
  "signature": {
    "collector_signed": ""
  }
}

Extraction Rules:
1. header.contact = ONLY phone number after "Tel" or "โทร", exclude fax
2. customer.name = The PAYER/COMPANY, NOT the guest name. Look for "Company", "คณะ", "บริษัท", or organization names. If there's both a "Guest's Name" and "Company", use the Company name.
3. customer.address = Company/payer address (look for "Company Address" field), keep in ORIGINAL language
4. customer.tax_id = Company/payer tax ID (look near company info or "เลขประจำตัวผู้เสียภาษี")
5. line_items.description = Combine ALL text for that item (product name + dates/details if any, but NOT the guest name)
6. payment.total = subtotal before VAT (or "Expenses")
7. payment.vat = VAT amount (0 if not shown)
8. payment.net_total = final total after VAT (or "Total Expenses")
9. signature.collector_signed = "Yes" if you see ANY handwriting, signature, or ink marks in the "Issued by/Collector/ออกโดย/ผู้รับเงิน/GUEST'S SIGNATURE/CASHIER'S SIGNATURE" sections, otherwise "No"
10. Keep all Thai text in Thai, all English in English - do NOT translate
11. Double-check Thai month abbreviations and characters (หมู่ not นม, เอกสาร not อกสาร)
12. date = Transaction date or check-out date or printed date (in format shown on receipt)"""
    
    response = client.chat.completions.create(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
            ]
        }],
        temperature=0
    )
    
    content = response.choices[0].message.content
    
    # Clean response
    content = re.sub(r'```json\s*', '', content)
    content = re.sub(r'```\s*', '', content)
    content = content.strip()
    
    # Clean up temp file
    import os
    if os.path.exists(preprocessed_path):
        os.remove(preprocessed_path)
    
    return json.loads(content)
