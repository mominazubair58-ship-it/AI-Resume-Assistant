import io
import json
import os

import streamlit as st
from docx import Document
from google import genai
from google.genai import types
from pypdf import PdfReader

MODEL_NAME = "gemini-2.5-flash"


def get_api_key():
    """Get Gemini API key from Streamlit Secrets or environment variables."""
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        api_key = None

    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured. Add it to "
            ".streamlit/secrets.toml or your environment variables."
        )

    return api_key


def extract_text(uploaded_file):
    """Extract text from PDF or DOCX files."""
    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if filename.endswith(".docx"):
        document = Document(io.BytesIO(file_bytes))
        return "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        )

    raise ValueError("Unsupported file type. Please upload a PDF or DOCX.")


def analyze_resume(resume_text, job_description=""):
    """Analyze a resume with Gemini and return structured JSON."""
    client = genai.Client(api_key=get_api_key())

    resume_text = resume_text[:30000]
    job_description = job_description[:20000]

    if job_description.strip():
        task_context = f"""
A job description was also provided. In this case:
- Calculate a job-specific ATS match score from 0 to 100.
- Compare the resume against the job description.
- Identify matched and missing keywords/skills.
- Do not claim a keyword is present unless it appears in the resume.

JOB DESCRIPTION:
{job_description}
"""
    else:
        task_context = """
No job description was provided.
Give a general ATS-readiness score from 0 to 100 based on:
- ATS-friendly structure
- standard resume sections
- readability
- skills and keywords
- action verbs
- measurable achievements
- consistency
Do not pretend this is a match score for a particular job.
"""

    prompt = f"""
You are an expert ATS resume analyzer and professional career coach.

Analyze the resume using only the information provided.

Rules:
1. Never invent experience, skills, education, projects, or achievements.
2. Give practical and specific recommendations.
3. Keep recommendations concise and useful.
4. A higher score means the resume is more ATS-ready / better matched.
5. Mention important weaknesses clearly.
6. If information is missing, say it is missing instead of guessing.

{task_context}

RESUME:
{resume_text}
"""

    schema = {
        "type": "OBJECT",
        "properties": {
            "ats_score": {"type": "INTEGER", "description": "Score from 0 to 100."},
            "score_type": {"type": "STRING", "description": "Either General ATS Readiness or Job Match."},
            "summary": {"type": "STRING", "description": "Short overall assessment."},
            "strengths": {"type": "ARRAY", "items": {"type": "STRING"}},
            "improvements": {"type": "ARRAY", "items": {"type": "STRING"}},
            "matched_keywords": {"type": "ARRAY", "items": {"type": "STRING"}},
            "missing_keywords": {"type": "ARRAY", "items": {"type": "STRING"}},
            "formatting_issues": {"type": "ARRAY", "items": {"type": "STRING"}},
        },
        "required": [
            "ats_score",
            "score_type",
            "summary",
            "strengths",
            "improvements",
            "matched_keywords",
            "missing_keywords",
            "formatting_issues",
        ],
    }

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0.2,
        ),
    )

    return json.loads(response.text)


def display_list(title, items, empty_message):
    st.subheader(title)
    if items:
        for item in items:
            st.write(f"• {item}")
    else:
        st.write(empty_message)


st.set_page_config(
    page_title="Resume ATS Analyzer",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Resume ATS Analyzer")
st.write(
    "Upload your resume to receive an AI-powered ATS score, "
    "strengths, weaknesses, keyword analysis, and improvement suggestions."
)

st.info(
    "Tip: For a job-specific ATS score, paste the Job Description below. "
    "Without one, the app gives a general ATS-readiness score."
)

uploaded_file = st.file_uploader(
    "Upload your resume",
    type=["pdf", "docx"],
    help="Supported formats: PDF and DOCX.",
)

job_description = st.text_area(
    "Job Description (optional)",
    height=220,
    placeholder="Paste the job description here for a job-specific ATS match score...",
)

if uploaded_file:
    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("Analyze Resume", type="primary", use_container_width=True):
        try:
            with st.spinner("Extracting resume text..."):
                resume_text = extract_text(uploaded_file)

            if not resume_text.strip():
                st.error(
                    "No readable text was found in this file. "
                    "If it is a scanned/image-only PDF, please use a text-based PDF or DOCX."
                )
                st.stop()

            with st.spinner("Analyzing your resume with Gemini..."):
                result = analyze_resume(resume_text, job_description)

        except Exception as error:
            st.error(f"Analysis failed: {error}")
            st.stop()

        score = max(0, min(100, int(result.get("ats_score", 0))))

        st.divider()
        st.subheader("🎯 ATS Score")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Score", f"{score}/100")

        with col2:
            if score >= 80:
                status = "Strong"
            elif score >= 60:
                status = "Needs Improvement"
            else:
                status = "Needs Major Improvement"
            st.metric("Status", status)

        with col3:
            st.metric("Analysis", result.get("score_type", "ATS Analysis"))

        st.progress(score / 100)

        st.subheader("📋 Overall Assessment")
        st.write(result.get("summary", ""))

        display_list(
            "✅ Strengths",
            result.get("strengths", []),
            "No major strengths were identified.",
        )

        display_list(
            "🔧 Recommended Improvements",
            result.get("improvements", []),
            "No major improvements were identified.",
        )

        display_list(
            "🔑 Matched Keywords",
            result.get("matched_keywords", []),
            "No matched keywords were identified.",
        )

        display_list(
            "⚠️ Missing / Weak Keywords",
            result.get("missing_keywords", []),
            "No obvious keyword gaps were identified.",
        )

        display_list(
            "📝 Formatting / ATS Issues",
            result.get("formatting_issues", []),
            "No major formatting issues were identified.",
        )

        st.divider()
        st.caption(
            "The ATS score is an AI-generated estimate. "
            "Different applicant tracking systems may score resumes differently."
        )
