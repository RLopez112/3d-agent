import fitz  # PyMuPDF
import ollama

def describe_first_pdf_page(pdf_path: str, model_name: str = 'llava'):
    """
    Extracts the first page of a PDF as an image and uses an Ollama vision model to describe it.
    """
    print(f"Opening '{pdf_path}'...")
    
    # 1. Open the PDF and grab the first page (index 0)
    try:
        doc = fitz.open(pdf_path)
        first_page = doc.load_page(0)
    except Exception as e:
        return f"Error loading PDF: {e}"

    # 2. Render the page to a pixmap (image)
    # The matrix increases the resolution for the vision model (scale factor of 2)
    zoom_matrix = fitz.Matrix(2, 2)
    pix = first_page.get_pixmap(matrix=zoom_matrix)
    
    # 3. Convert the image to bytes
    image_bytes = pix.tobytes("png")
    doc.close()

    print(f"Sending page 1 to Ollama model '{model_name}'...\n")

    # 4. Send the image bytes and prompt to Ollama
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{
                'role': 'user',
                'content': 'Describe the images of this PDF page.',
                'images': [image_bytes]
            }]
        )
        return response['message']['content']
    
    except Exception as e:
        return f"Error communicating with Ollama: {e}"

# --- Execution ---
if __name__ == "__main__":
    pdf_file = "/home/rodrigo/repos/3d-agent/assets/project-assets.pdf" 
    description = describe_first_pdf_page(pdf_file)
    
    print("--- Model Description ---")
    print(description)