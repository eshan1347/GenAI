from pypdf import PdfReader 
import os
import chromadb
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

pdf_directory = 'PDFs'
output_file = 'text.txt'

# Extracting Text from all pdfs present in 'PDFs' folder

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

# Extract text from all PDFs in the directory and save to a single file
with open(output_file, 'w') as output:
    for pdf_file in os.listdir(pdf_directory):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_directory, pdf_file)
            pdf_text = extract_text_from_pdf(pdf_path)
            output.write(f"--- Start of {pdf_file} ---\n")
            output.write(pdf_text)
            output.write(f"\n--- End of {pdf_file} ---\n\n")

# Extracting text from all links present on a page and then parsing the text from html 

def fetchURL(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.text

def extractLinks(html, url):
    soup = BeautifulSoup(html, 'html.parser')
    links = []
    for l in soup.find_all('a', href=True):
        link = l['href']
        fLink = urljoin(url, link)
        links.append(fLink)
    return links

def extractText(html):
    soup = BeautifulSoup(html, 'html.parser')
    # text = soup.get_text()
    text = ""
    main = soup.find(id="main-content")
    if main:
        text = main.get_text()
    # paragraphs = text.split('\n\n')  # Split by paragraphs; adjust if needed
    return text

def getData(url):
    base_html = fetchURL(url)
    links = extractLinks(base_html,url)
    text = ""
    for l in links:
        html = fetchURL(l)
        text += extractText(html)
    return text

data = getData('https://www.apple.com/apple-vision-pro/')
# filename = 'text.txt'
with open(output_file, "a", encoding="utf-8") as file:
    file.write(data)
