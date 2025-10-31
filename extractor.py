import base64
import json
import re
from groq import Groq

def extract_receipt_data(image_path, groq_api_key):
    client = Groq(api_key=groq_api_key)
    
    # Read and encode image
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

Rules:
- header.tax_id = seller/merchant/service provider tax ID (near merchant info)
- customer.tax_id = buyer/customer tax ID (near customer info)
- If only one tax ID exists, put it in header.tax_id and leave customer.tax_id empty
- customer fields can be empty if not present"""
    
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
    
    # Clean response - remove markdown code blocks if present
    content = re.sub(r'```json\s*', '', content)
    content = re.sub(r'```\s*', '', content)
    content = content.strip()
    
    return json.loads(content)
