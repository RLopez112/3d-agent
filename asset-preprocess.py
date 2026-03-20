import os
from PIL import Image
import io
import PyPDF2
import requests
import cv2
import numpy as np
import base64

def load_pdf(file_path):
    try:
        
        reader = PyPDF2.PdfReader(file_path)
        return reader
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def extract_images_from_pdf(file_path):
             #save the first image from the PDF to a file
    with open(os.path.join("/home/rodrigo/repos/3d-agent", "a.png"), "wb") as fp:
                fp.write(load_pdf(file_path).pages[0].images[0].data) #

def extract_text_from_pdf(file_path):
         pdf_text = load_pdf(file_path).pages[0].extract_text()  # Extract text from the first page
         return pdf_text

def encode_image(image_path):
        """Convierte una imagen local en una cadena base64."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

def ollama_pdf_refview(pdf_path):
    api_endpoint = 'http://localhost:11434/api/generate'
    


  
    
    # 2. Prepara la imagen
    path_to_image = "/home/rodrigo/repos/3d-agent/a.png" # Cambia esto por la ruta de tu archivo

    base64_image = encode_image(path_to_image)

 
    data = {
        'model': 'qwen3.5:4b',
        'stream': False,
        'prompt': f'what is this?: \n\nImage: {base64_image}'
        #\n\n{pdf_text}
    }

    headers = {
        'Content-Type': 'application/json',
            'Authorization': f'bearer {os.getenv("OLLAMA_API_KEY")}'  # Replace with your actual API key
    }

    response = requests.post(api_endpoint, json=data, headers=headers)

    if response.status_code == 200:
        response_data = response.json()
        print(response_data['response'])
    else:
        print('Failed to get response from Ollama API. Status code:', response.status_code)
        print('Response text:', response.text) # Helpful for debugging


def process_all_pdfs(folder_path):
    """Find all PDFs in the specified folder and run ollama_pdf_refview on each one."""
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]

    for file_name in pdf_files:
        file_path = os.path.join(folder_path, file_name)
        ollama_pdf_refview(file_path)




#process_all_pdfs(folder_path)



#ollama_pdf_refview("project-assets.pdf")
print(encode_image("/home/rodrigo/repos/3d-agent/a.png"))

