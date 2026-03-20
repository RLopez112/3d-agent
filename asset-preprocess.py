from PyPDF2 import PdfFileReader
import requests


def load_pdf(file_path):
    try:
        pdf_file = open(file_path, 'rb')
        reader = PdfFileReader(pdf_file)
        return reader
    except Exception as e:
        print(f"An error occurred: {str(e)}")



def ollama_pdf_refview():
    api_endpoint = 'http://localhost:11434/api/generate'

    data = {
        'model' : 'llama3:8b',
        'stream' : False,
        'prompt' : 'Act as a translator machine. Translate this to spanish: car'
    }

    headers = {
        'Content-Type': 'application/json',
            'Authorization': 'bearer YOUR_OLLAMA_API_KEY'  # Replace with your actual API key
    }

    response = requests.post(api_endpoint, json=data, headers=headers)

    if response.status_code == 200:
        response_data = response.json()
        print(response_data['response'])
    else:
        print('Failed to get response from Ollama API. Status code:', response.status_code)
        print('Response text:s', response.text) # Helpful for debugging

    # Call the new function
ollama_pdf_refview()
