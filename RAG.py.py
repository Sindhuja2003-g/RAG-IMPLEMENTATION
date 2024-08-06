from PyPDF2 import PdfReader
import chromadb
import google.generativeai as genai
from google.colab import userdata

def split_pdf_to_text_list(input_pdf_path):
    # Create a PDF reader object
    reader = PdfReader(input_pdf_path)
    num_pages = len(reader.pages)

    # List to store text content of each page
    pages_content = []

    # Iterate through all the pages
    for i in range(num_pages):
        page = reader.pages[i]
        text = page.extract_text()

        # Append the text content to the list
        pages_content.append(text)

    return pages_content


input_pdf_path = "/content/HDFC-Life-Sanchay-Plus-Life-Long-Income-Option-101N134V19-Policy-Document.pdf"
pages_content = split_pdf_to_text_list(input_pdf_path)


chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="pdf_pages")

collection.add(
    documents=pages_content,
    ids=[str(i) for i in range(len(pages_content))]
)

collection.get(include=['embeddings'])

user_input=input()

context = collection.query(
    query_texts=user_input, # Chroma will embed this for you
    n_results=1 # how many results to return
)
context

prompt = f"Based on following context, answer the query:\nContext:{context}\nQuery:{user_input}\nAnswer:"

GOOGLE_API_KEY=userdata.get('geminiapi')
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-pro')

response = model.generate_content(prompt,generation_config={"max_output_tokens":100,"temperature":0.2})
print(response.text)