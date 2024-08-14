import streamlit as st
import io
import base64
from PIL import Image as PILImage
from typing import Any
from pydantic import BaseModel
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Image as UnstructuredImage
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import os
import tempfile
import logging
import google.generativeai as genai

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Gemini model
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
vision_model = genai.GenerativeModel('gemini-pro-vision')
assessment_model = genai.GenerativeModel('gemini-pro')

def partition_pdf_file(file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.getvalue())
            temp_file_path = temp_file.name

        logger.info(f"Partitioning PDF: {temp_file_path}")
        rpe = partition_pdf(
            filename=temp_file_path,
            extract_images_in_pdf=True,
            infer_table_structure=True,
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000
        )
        logger.info(f"Partition complete. Number of elements: {len(rpe)}")

        return rpe, os.path.dirname(temp_file_path)
    except Exception as e:
        st.error(f"Error in partition_pdf_file: {str(e)}")
        logger.error(f"Error in partition_pdf_file: {str(e)}", exc_info=True)
        return None, None

def image2base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        logger.info(f"Extracted text length: {len(text)}")
        return text
    except Exception as e:
        st.error(f"Error in extract_text_from_pdf: {str(e)}")
        logger.error(f"Error in extract_text_from_pdf: {str(e)}", exc_info=True)
        return ""


def convert_handwriting_to_text(image_path):
    image = PILImage.open(image_path)
    
    prompt = """Convert the provided handwritten student answer sheets into a text document, ensuring precise and faithful transcription. The final output should follow these detailed guidelines:

    1. **Question Identification and Continuation:** 
       - Begin each answer with its corresponding question number.
       - If you did not find question number on the starting of next page, it means it is the continuation of previous answer, it should be correctly identified and formatted without repeating the question number. Use the notation '[continued]' after the partial answer and continue the rest seamlessly.

    2. **Exact Transcription:** 
       - Transcribe all written content exactly as it appears, preserving the student's original wording, spelling, and grammar. Avoid any interpretation or addition of content not present in the handwritten text.

    3. **Handling Incomplete Answers:** 
       - If an answer seems incomplete, indicate it with a note such as "[Note: The answer appears to be incomplete.]"
       - Ensure that continuation parts are not mistakenly attributed to incorrect question numbers.

    4. **Diagrams and Visual Elements:** 
       - Provide a detailed description of any diagrams, images, or charts included by the student. Describe the elements accurately without interpretation, ensuring clarity and completeness.

    5. **Formatting and Structure:** 
       - Maintain the original structure and layout, including bullet points, numbering, headings, and indentation. Ensure that each answer follows a consistent format with the question number at the beginning.

    6. **Legibility Issues:** 
       - If certain parts of the handwriting are unclear or illegible, indicate this with the notation "[illegible]".

    7. **No Additions or Hallucinations:** 
       - The transcription should be an accurate and faithful representation of the student's original work, without adding any external information or embellishments.

    Ensure the final output adheres to these guidelines, providing a clear, accurate, and well-organized transcription of the handwritten answer sheets."""

    response = vision_model.generate_content([prompt, image])
    return response.text

def assess_answers(student_answers, answer_key):
    prompt = f"""
    Compare the student's answers with the answer key and assign marks.
    
    Student Answers:
    {student_answers}
    
    Answer Key:
    {answer_key}
    
    Instructions:
    1. Analyze each answer provided by the student.
    2. Compare it with the corresponding answer in the answer key.
    3. Assign a score for each answer in the format 'question_number. score/total_mark'.
    4. If an answer is not provided, mark it as 'question_number. answer not provided'.
    5. Provide a brief explanation for each score given.
    6. Important at the end, calculate and display the total score secured by the student in the format 'Total: [total_scored]/[grand_total]'.
    
    Please provide your assessment:
    """
    
    response = assessment_model.generate_content(prompt)
    return response.text

st.title("Evaluate Answers")

# File uploaders
student_answer_file = st.file_uploader("Upload Student Answer Sheet (PDF)", type="pdf")
answer_key_file = st.file_uploader("Upload Answer Key (PDF)", type="pdf")

if student_answer_file and answer_key_file:
    if st.button("Evaluate"):
        with st.spinner("Processing..."):
            # Extract text from answer key PDF
            answer_key_text = extract_text_from_pdf(answer_key_file)

            # Partition the student answer sheet PDF
            rpe, temp_dir = partition_pdf_file(student_answer_file)
            
            if rpe and temp_dir:
                converted_text = ""
                
                # Process each page/image in the student answer sheet
                for i, element in enumerate(rpe):
                    if isinstance(element, UnstructuredImage):
                        
                        if hasattr(element.metadata, 'image_path'):
                            image_path = os.path.join(temp_dir, element.metadata.image_path)

                            if os.path.exists(image_path):
                                try:
                                    # Display the image
                                    image = PILImage.open(image_path)
                                    # st.image(image, caption=f"Student Answer Sheet - Page {i+1}", use_column_width=True)

                                    # Convert handwriting to text
                                    image_base64 = image2base64(image_path)
                                    page_text = convert_handwriting_to_text(image_base64)
                                    converted_text += page_text + "\n\n"

                                except Exception as e:
                                    st.error(f"Error processing image: {str(e)}")
                                    logger.error(f"Error processing image: {str(e)}", exc_info=True)
                            else:
                                st.write(f"Image file not found: {image_path}")
                        else:
                            st.write(f"Element {i+1} does not contain image_path in metadata")
                    else:
                        st.write(f"Element {i+1} is not an Image element")

                # Save converted text to file
                with open("student_answers.txt", "w") as f:
                    f.write(converted_text)
                
                # st.text(converted_text)

                # Assess answers
                assessment = assess_answers(converted_text, answer_key_text)
                # st.write("Assessment:")
                st.write(assessment)

            else:
                st.error("Failed to partition the PDF")
else:
    st.warning("Please upload both the Student Answer Sheet and the Answer Key PDFs.")