import fitz  # PyMuPDF
import re  # regex for text extraction


def extract_text_from_pdf(path):
    doc = fitz.open(path)
    return "".join([page.get_text() for page in doc])


def extract_skills(text):
    keywords = ["python", "java", "sql", "html", "css", "excel", "ml", "ai"]
    return [kw for kw in keywords if kw in text.lower()]


def extract_name(text):
    # Try regex near the word "Name" or assume first non-empty line
    match = re.search(r"(?i)name[:\-]?\s*([A-Z][a-z]+\s[A-Z][a-z]+)", text)
    if match:
        return match.group(1)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else ""


def extract_degree(text):
    degrees = [
        "bachelor of science", "b.sc", "b.s", "bachelor of engineering", "b.e", "b.tech",
        "master of science", "m.sc", "m.s", "m.tech", "mba", "phd", "mca", "bca"
    ]
    text_lower = text.lower()
    found = [deg for deg in degrees if deg in text_lower]
    return found if found else []


def extract_experience(text):
    # Look for patterns like "X years of experience"
    match = re.search(r"(\d+)\s+\+?\s*(?:years|yrs)\s+of\s+experience", text, re.IGNORECASE)
    if match:
        return match.group(1) + " years"
    # Try another common phrasing
    match = re.search(r"experience\s*[:\-]?\s*(\d+)\s+\+?\s*(?:years|yrs)", text, re.IGNORECASE)
    if match:
        return match.group(1) + " years"
    return "Not specified"


def parse_resume(file_path):
    text = extract_text_from_pdf(file_path)
    return {
       # "name": extract_name(text),
        "skills": extract_skills(text),
        "degree": extract_degree(text),
        "experience": extract_experience(text)
    }


# Example
#print(parse_resume("sampleresume3.pdf"))
