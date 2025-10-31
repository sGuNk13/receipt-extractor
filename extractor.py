import base64
import json
from groq import Groq

def extract_receipt_data(image_path, groq_api_key):
    client = Groq(api_key=groq_api_key)
    
    # Read and encode image
    with open(image_path, "rb") as img_file:
        image_data = base64.b64encode(img_file.read()).decode('utf-8')
    
    prompt = """Extract these fields from the receipt image:

Header:
- Merchant name
- Address
- Tax ID
- Contact

Date:
- Transaction date

Line items (list each):
- Description
- Amount

Payment:
- Total
- VAT (if available, else null)
- Net total

Signature:
- Collector signed (Yes/No - check if there's any signature)

Return valid JSON only, no explanation."""
    
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
            ]
        }],
        temperature=0
    )
    
    return json.loads(response.choices[0].message.content)
