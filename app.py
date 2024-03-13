import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import re
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
import pytesseract
from pytesseract import Output
from PIL import Image
from flask import Flask, render_template, request
import PyPDF2

# nltk.download('words')
# nltk.download('punkt')
# nltk.download('stopwords')

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    words = word_tokenize(text)
    english_vocab = set(word.lower() for word in nltk.corpus.words.words())
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    text = ' '.join(words)
    return text

def divide_documents(text):
    paragraphs = text.split('\n\n')
    sentences = []
    for paragraph in paragraphs:
        sentences.extend(sent_tokenize(paragraph))
    return sentences

def generate_summaries(sentences):
    summaries = []
    for sentence in sentences:
        summary = sentence
        summaries.append(summary)
    return summaries

def readable_summary(text):
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    input_ids = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(input_ids, max_length=150, num_beams=4, length_penalty=2.0, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", tokenizer="distilbert-base-cased-distilled-squad")

# Specify the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)

# nltk.download('words')
# nltk.download('punkt')
# nltk.download('stopwords')

def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PdfFileReader(pdf_file)
    for page_number in range(pdf_reader.numPages):
        page = pdf_reader.getPage(page_number)
        text += page.extractText()
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_file', methods=['POST'])
def process_file():
    # Get the choice of the user (image or pdf)
    file_type = request.form['file_type']

    if file_type == 'image':
        # Example: Get the uploaded image file
        uploaded_file = request.files['file']

        # Perform OCR on the image
        with Image.open(uploaded_file) as image:
            txt = pytesseract.image_to_string(image)

    elif file_type == 'pdf':
        # Example: Get the uploaded PDF file
        uploaded_file = request.files['file']

        # Extract text from the PDF using PyPDF2
        pdf_reader = PyPDF2.PdfReader(uploaded_file)

        # Get the number of pages in the PDF
        num_pages =num_pages = len(pdf_reader.pages)

        # Initialize an empty string to store the extracted text
        txt = ""

        # Iterate through all pages
        for page_num in range(num_pages):
            # Get the page
            page = pdf_reader.pages[page_num]

            # Extract text from the page
            etext = page.extract_text()

            # Append the text to the result string
            txt+=etext

    sample_text = txt
    # Preprocess text using the loaded function
    preprocessed_text = preprocess_text(sample_text)

    # Divide documents into sentences using the loaded function
    sentences = divide_documents(preprocessed_text)

    # Generate summaries using the loaded function
    summaries = generate_summaries(sentences)

    # Check if the user wants to view the summary
    show_summary = request.form.get('show_summary')

    if show_summary:
        # Use the loaded function to generate a readable summary
        summary = readable_summary(sample_text)
    else:
        summary = None

    # Example: Get the question from the user
    user_question = request.form['question']

    # Use the loaded QA pipeline
    answer = qa_pipeline(question=user_question, context=sample_text)

    # Render the result on the webpage
    return render_template('result.html', question=user_question, answer=answer['answer'], summary=summary)

if __name__ == '__main__':
    app.run(debug=True)

