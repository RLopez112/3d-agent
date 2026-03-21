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


#ollama_pdf_refview("project-assets.pdf")
print(encode_image("/home/rodrigo/repos/3d-agent/a.png"))

