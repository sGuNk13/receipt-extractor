import base64
import json
import re
from groq import Groq

# Common Thai words dictionary for spell-check
THAI_COMMON_WORDS = {
    "ค่าออกธรบ": "ค่าเอกสาร",
    "ค่าอกสาร": "ค่าเอกสาร",
    "บัณฑิต": "และการบัญชี",
    "นม": "หมู่",
}

def spell_check_thai(text):
    """Basic spell check for common Thai OCR errors"""
    for wrong, correct in THAI_COMMON_WORDS.items():
        text = text.replace(wrong, correct)
    return text

def extract_receipt_data(image_path, groq_api_key):
    client = Groq(api_key=groq_api_key)
    
    # Read and encode image WITHOUT preprocessing for handwritten text
    with open(image_path, "rb") as img_file:
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
2. customer.name = If both "Company" AND "Guest's Name" fields exist, use the Company name (not guest). Otherwise use the customer/buyer name shown.
3. customer.address = Keep in ORIGINAL language (Thai or English as shown)
4. line_items.description = Combine ALL text for that item. Common items: ค่าเอกสาร (documents), ห้องพัก (room), อาหาร (food)
5. payment.total = subtotal before VAT
6. payment.vat = VAT amount (0 if not shown)
7. payment.net_total = final total after VAT
8. signature.collector_signed = "Yes" if you see ANY handwriting, signature, or ink marks in the "Issued by/Collector/ออกโดย/ผู้รับเงิน/CASHIER'S SIGNATURE" section, otherwise "No"
9. Keep all Thai text in Thai, all English in English - do NOT translate
10. Thai months: ม.ค.=Jan, ก.พ.=Feb, มี.ค.=Mar, เม.ย.=Apr, พ.ค.=May, มิ.ย.=Jun, ก.ค.=Jul, ส.ค.=Aug, ก.ย.=Sep, ต.ค.=Oct, พ.ย.=Nov, ธ.ค.=Dec
11. BE VERY CAREFUL with Thai year (พ.ศ.): 2568 is NOT 2562, 2567 is NOT 2565. Read the numbers precisely."""
    
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
    
    # Apply spell check
    content = spell_check_thai(content)
    
    return json.loads(content)
