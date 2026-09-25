import os
import json
import re
import time
import shutil
import asyncio
import base64
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
import fitz  # PyMuPDF: fast local extraction for text PDFs

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# Vercel serverless functions have a read-only project filesystem.
# /tmp is the writable temporary directory.
UPLOAD_DIR = Path("/tmp/bloomai_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing in .env")

# Fast Groq models. Text/analysis uses GPT-OSS 20B; vision is used only for scanned PDFs.
FAST_MODEL = "openai/gpt-oss-20b"
VISION_MODEL = "qwen/qwen3.8-27b"
STT_MODEL = "whisper-large-v3-turbo"

# Short timeout and no client-side retry chain: the UI should fail quickly instead of hanging.
client = Groq(api_key=GROQ_API_KEY, timeout=20.0, max_retries=0)

app = FastAPI(title="BloomAI", version="4.0")

last_uploaded_file = None
last_analysis: dict[str, Any] | None = None
last_filename = ""


class AskRequest(BaseModel):
    question: str
    mode: str = "normal"
    marks: int | None = None
    bloom_level: str = ""
    difficulty: str = ""
    topic: str = ""


class QuizRequest(BaseModel):
    count: int = 10


class StudyRequest(BaseModel):
    focus: str = "all"


ANALYSIS_PROMPT = r'''
You are BloomAI, an academic question-paper analysis agent.

Read EVERY page of the uploaded PDF, including scanned/image-based pages. Visually inspect the pages. The document may contain one or multiple examination papers from different years.

Return ONLY valid JSON. Use exactly this structure:
{
  "document_type":"",
  "subject":"",
  "paper_count":1,
  "papers":[{"label":"","year":"","max_marks":0,"question_count":0}],
  "total_questions":0,
  "total_marks":0,
  "average_difficulty":"Moderate",
  "average_difficulty_score":50,
  "paper_quality_score":0,
  "bloom_distribution":{"Remember":0,"Understand":0,"Apply":0,"Analyze":0,"Evaluate":0,"Create":0},
  "difficulty_distribution":{"Easy":0,"Moderate":0,"Difficult":0,"Very Difficult":0},
  "topic_distribution":[{"topic":"","count":0}],
  "questions":[{"question":"","marks":0,"paper":"","question_number":"","bloom_level":"","difficulty":"","difficulty_score":0,"topics":[],"answer":""}],
  "important_questions":[{"question":"","reason":"","answer":"","marks":0,"paper":"","bloom_level":"","difficulty":""}],
  "insights":[],
  "issues":[],
  "study_priorities":[]
}

Rules:
1. Extract every visible academic question. Never invent a question.
2. Preserve wording as closely as possible. If unreadable, say so rather than guessing.
3. Bloom levels: Remember, Understand, Apply, Analyze, Evaluate, Create.
4. Difficulty: Easy, Moderate, Difficult, Very Difficult. Score 0-100.
5. Match answer length to marks. Keep 2-mark answers concise; make higher-mark answers structured and detailed.
6. Detect separate papers by year/exam when possible.
7. total_marks should use clearly printed maximum marks for each detected paper, summed across papers. If unavailable, explain the fallback in insights.
8. Group recurring topics in topic_distribution.
9. Detect repeated/near-repeated questions across years.
10. paper_quality_score is based on clarity, coverage, marks balance, Bloom balance, duplication and ambiguity.
11. important_questions must contain the most useful/repeated/high-mark/difficult questions supported by this document, with exam-ready answers.
12. Every question object must have a non-empty exam-ready answer matched to its marks and wording. Never leave answer as an empty string.
13. study_priorities must be concrete topics or question groups to revise.
14. Do not make claims about the institution or examiner.
15. JSON only. No markdown fences.
'''


def parse_json(text: str):
    text = (text or "").strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.I)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("AI did not return valid JSON.")
    return json.loads(text[start:end + 1])


def parse_json_array(text: str):
    text = (text or "").strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.I)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("AI did not return a valid JSON array.")
    return json.loads(text[start:end + 1])


def error_message(exc: Exception) -> str:
    s = str(exc)
    low = s.lower()
    if "401" in low or "invalid_api_key" in low or "authentication" in low:
        return "Groq API key was rejected. Check GROQ_API_KEY in .env."
    if "429" in low or "rate_limit" in low:
        return "Groq rate limit reached. Please wait a moment and try again."
    if "timeout" in low or "timed out" in low:
        return "Groq took too long to respond. Please try again."
    if "503" in low or "502" in low or "500" in low or "unavailable" in low:
        return "Groq is temporarily unavailable. Please try again."
    return s


def _prompt_from_contents(contents):
    """Convert the old Gemini-style contents argument into plain text for Groq."""
    if isinstance(contents, str):
        return contents
    if isinstance(contents, (list, tuple)):
        parts = []
        for item in contents:
            if isinstance(item, str):
                parts.append(item)
        return "\n\n".join(parts)
    return str(contents)


def generate_text(contents, *, smart=False, attempts=1, max_output_tokens=1800):
    prompt = _prompt_from_contents(contents)
    model = FAST_MODEL
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are BloomAI, a fast academic assistant. Follow the user's requested format exactly."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_completion_tokens=max_output_tokens,
    )
    text = completion.choices[0].message.content if completion.choices else ""
    if not text:
        raise RuntimeError("Groq returned an empty response.")
    return text, model


def stream_text(contents, *, max_output_tokens=1000):
    prompt = _prompt_from_contents(contents)
    stream = client.chat.completions.create(
        model=FAST_MODEL,
        messages=[
            {"role": "system", "content": "You are BloomAI, a fast academic assistant. Answer directly and concisely."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_completion_tokens=max_output_tokens,
        stream=True,
    )
    sent = False
    for chunk in stream:
        if not chunk.choices:
            continue
        text = chunk.choices[0].delta.content or ""
        if text:
            sent = True
            yield text
    if not sent:
        raise RuntimeError("Groq returned an empty response.")


def extract_pdf_text(file_path: Path) -> str:
    with fitz.open(file_path) as pdf:
        pages = []
        for page in pdf:
            txt = page.get_text("text") or ""
            if txt.strip():
                pages.append(txt.strip())
        return "\n\n".join(pages).strip()


def extract_scanned_pdf_text(file_path: Path) -> str:
    """Use Groq vision OCR for image-only PDFs, in small page batches."""
    chunks = []
    with fitz.open(file_path) as pdf:
        for start in range(0, len(pdf), 3):
            content = [{
                "type": "text",
                "text": "Read these question-paper page images. Transcribe ALL visible text accurately, including headings, question numbers, marks, options and tables. Return plain text only. Do not invent unreadable text."
            }]
            for page_index in range(start, min(start + 3, len(pdf))):
                page = pdf[page_index]
                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                image_b64 = base64.b64encode(pix.tobytes("jpeg", jpg_quality=82)).decode("ascii")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                })
            response = client.chat.completions.create(
                model=VISION_MODEL,
                messages=[{"role": "user", "content": content}],
                temperature=0,
                max_completion_tokens=5000,
            )
            text = response.choices[0].message.content if response.choices else ""
            if text:
                chunks.append(text)
    return "\n\n".join(chunks).strip()


def transcribe_audio_bytes(data: bytes, filename: str = "voice.webm") -> str:
    if not data:
        raise RuntimeError("No audio was recorded.")
    if len(data) > 25 * 1024 * 1024:
        raise RuntimeError("Voice recording is too large. Please keep it shorter.")
    result = client.audio.transcriptions.create(
        file=(filename, data),
        model=STT_MODEL,
        language="en",
        response_format="json",
        temperature=0.0,
    )
    text = getattr(result, "text", "") or ""
    if not text.strip():
        raise RuntimeError("No speech was detected.")
    return text.strip()

def fill_missing_answers(analysis: dict[str, Any], source_text: str) -> dict[str, Any]:
    """Fill only missing/empty answers using the uploaded paper's questions and source text."""
    questions = analysis.get("questions") or []
    missing = []
    for idx, q in enumerate(questions):
        if not str(q.get("answer") or "").strip():
            missing.append({
                "index": idx,
                "question": q.get("question", ""),
                "marks": q.get("marks", 0),
                "bloom_level": q.get("bloom_level", ""),
                "difficulty": q.get("difficulty", ""),
                "topics": q.get("topics", []),
            })
    if not missing:
        answer_map = {str(q.get("question", "")).strip(): q.get("answer", "") for q in questions}
        for item in analysis.get("important_questions") or []:
            if not str(item.get("answer") or "").strip():
                item["answer"] = answer_map.get(str(item.get("question", "")).strip(), "")
        return analysis

    source = (source_text or "")[:70000]
    prompt = f"""You are repairing missing exam-ready answers in BloomAI.
Use ONLY the uploaded question-paper text below as the source for terminology, scope and question context.
For each listed question, write an accurate answer appropriate to its marks and Bloom level.
Do not invent a different question, topic, institution, year or unsupported source-specific fact.
If the source does not provide a source-specific fact needed by the question, answer the academic question using standard subject knowledge, while staying faithful to the question wording.
Return ONLY a JSON array. Each item must be {{"index": number, "answer": "string"}}.
Every answer must be non-empty.

UPLOADED PAPER TEXT:
{source}

QUESTIONS REQUIRING ANSWERS:
{json.dumps(missing, ensure_ascii=False)}"""
    try:
        text, _ = generate_text(prompt, smart=False, attempts=1, max_output_tokens=9000)
        repaired = parse_json_array(text)
        by_index = {int(x.get("index")): str(x.get("answer") or "").strip() for x in repaired if isinstance(x, dict) and str(x.get("answer") or "").strip()}
        for idx, q in enumerate(questions):
            if not str(q.get("answer") or "").strip() and idx in by_index:
                q["answer"] = by_index[idx]
    except Exception:
        pass

    answer_map = {str(q.get("question", "")).strip(): q.get("answer", "") for q in questions}
    for item in analysis.get("important_questions") or []:
        if not str(item.get("answer") or "").strip():
            item["answer"] = answer_map.get(str(item.get("question", "")).strip(), "")
    return analysis


def _derive_paper_difficulty(data: dict[str, Any]) -> dict[str, Any]:
    """Derive overall paper difficulty from the analyzed questions, not the model default."""
    questions = data.get("questions") or []
    label_mid = {
        "Easy": 25,
        "Moderate": 50,
        "Difficult": 70,
        "Very Difficult": 90,
    }
    scores = []
    distribution = {k: 0 for k in label_mid}

    for q in questions:
        label = str(q.get("difficulty") or "").strip()
        raw = q.get("difficulty_score", None)
        try:
            score = float(raw)
        except (TypeError, ValueError):
            score = None

        # Gemini sometimes emits the schema default 50 even when it classified
        # the question as Easy/Difficult/Very Difficult. In that case use the
        # classification's representative score instead of treating 50 as real.
        if label in label_mid:
            if score is None or (score == 50 and label != "Moderate"):
                score = float(label_mid[label])
            distribution[label] += 1
        elif score is not None:
            if score < 40:
                inferred = "Easy"
            elif score < 60:
                inferred = "Moderate"
            elif score < 80:
                inferred = "Difficult"
            else:
                inferred = "Very Difficult"
            distribution[inferred] += 1

        if score is not None:
            scores.append(max(0.0, min(100.0, score)))

    if not scores:
        return data

    avg = sum(scores) / len(scores)
    if avg < 40:
        overall = "Easy"
    elif avg < 60:
        overall = "Moderate"
    elif avg < 80:
        overall = "Difficult"
    else:
        overall = "Very Difficult"

    data["average_difficulty_score"] = round(avg)
    data["average_difficulty"] = overall
    if any(distribution.values()):
        data["difficulty_distribution"] = distribution
    return data


def normalize_analysis(data: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "document_type": "Question Paper",
        "subject": "Academic Question Paper",
        "paper_count": 1,
        "papers": [],
        "total_questions": 0,
        "total_marks": 0,
        "average_difficulty": "Moderate",
        "average_difficulty_score": 50,
        "paper_quality_score": 0,
        "bloom_distribution": {"Remember": 0, "Understand": 0, "Apply": 0, "Analyze": 0, "Evaluate": 0, "Create": 0},
        "difficulty_distribution": {"Easy": 0, "Moderate": 0, "Difficult": 0, "Very Difficult": 0},
        "topic_distribution": [],
        "questions": [],
        "important_questions": [],
        "insights": [],
        "issues": [],
        "study_priorities": []
    }
    for key, value in defaults.items():
        data.setdefault(key, value)
    data["questions"] = data.get("questions") or []
    data["total_questions"] = len(data["questions"])
    data = _derive_paper_difficulty(data)
    return data


@app.get("/", response_class=HTMLResponse)
async def home():
    return HTMLResponse(HTML_PAGE)


@app.get("/health")
async def health():
    return {"success": True, "status": "running", "fast_model": FAST_MODEL, "vision_model": VISION_MODEL, "stt_model": STT_MODEL}


@app.get("/test-ai")
async def test_ai():
    try:
        text, model = await asyncio.to_thread(generate_text, "Reply with exactly: BloomAI connection successful.", smart=False, attempts=1, max_output_tokens=100)
        return {"success": True, "message": text.strip(), "model": model}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"success": False, "message": error_message(exc)})


@app.post("/analyze")
async def analyze_pdf(file: UploadFile = File(...)):
    global last_uploaded_file, last_analysis, last_filename
    if not file.filename:
        raise HTTPException(400, "No file selected.")
    if Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(400, "Please upload a PDF question paper.")
    safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", file.filename)
    file_path = UPLOAD_DIR / safe_name
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        size = file_path.stat().st_size
        if size == 0:
            raise RuntimeError("The uploaded PDF is empty.")
        if size > 50 * 1024 * 1024:
            raise RuntimeError("PDF is larger than 50 MB. Please upload a smaller paper.")

        # FAST PATH: extract text locally. Image-only PDFs use Groq vision OCR in small batches.
        extracted_text = extract_pdf_text(file_path)
        if len(extracted_text) < 250:
            extracted_text = extract_scanned_pdf_text(file_path)

        if len(extracted_text) < 50:
            raise RuntimeError("Could not read enough text from this PDF. Please upload a clearer question paper.")

        fast_prompt = ANALYSIS_PROMPT + (
            "\n\nDOCUMENT TEXT EXTRACTED FROM THE PDF:\n"
            + extracted_text
        )
        ai_text, model_used = generate_text(fast_prompt, smart=False, attempts=1, max_output_tokens=10000)
        analysis = normalize_analysis(parse_json(ai_text))
        analysis = fill_missing_answers(analysis, extracted_text)
        # Keep the extracted paper text available for Paper mode without re-uploading the PDF.
        last_uploaded_file = extracted_text
        last_analysis = analysis
        last_filename = file.filename
        return {"success": True, "filename": file.filename, "model": model_used, "analysis": analysis}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"success": False, "message": error_message(exc)})


@app.post("/transcribe")
async def transcribe_voice(audio: UploadFile = File(...)):
    try:
        data = await audio.read()
        text = await asyncio.to_thread(transcribe_audio_bytes, data, audio.filename or "voice.webm")
        return {"success": True, "text": text}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"success": False, "message": error_message(exc)})


@app.post("/ask")
async def ask_bloomai(request: AskRequest):
    """General AI assistant by default.

    The user can explicitly choose mode=paper when they want answers grounded
    in the uploaded question paper. Speech-to-text feeds this same endpoint,
    so voice questions are not restricted to the uploaded document.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(400, "Please enter or speak a question.")
    if len(question) > 3000:
        raise HTTPException(400, "Question is too long.")

    try:
        mode = (request.mode or "general").lower()
        paper_mode = mode.startswith("paper")
        answer_mode = mode.replace("paper_", "", 1) if paper_mode else mode

        mode_instruction = {
            "short": "Give a concise answer.",
            "5mark": "Give a structured academic answer suitable for about 5 marks.",
            "10mark": "Give a detailed structured academic answer suitable for about 10 marks.",
            "simple": "Explain in simple student-friendly language and include a small example when useful.",
            "important": "Explain the concept clearly and highlight the most important points for study.",
            "paper": "Answer using the uploaded question paper as the primary source. If the requested information is not present in the uploaded paper, say so clearly rather than inventing it.",
            "general": "Answer as a general-purpose AI academic assistant. Do not assume the user is asking about the uploaded paper.",
            "normal": "Answer as a general-purpose AI academic assistant. Do not assume the user is asking about the uploaded paper."
        }.get(answer_mode, "Answer as a general-purpose AI academic assistant.")

        prompt = f"""You are BloomAI, a helpful general-purpose AI academic assistant.

{mode_instruction}

The user may ask about studying, programming, mathematics, technology, writing, general knowledge, concepts, or everyday academic questions. Answer the actual request directly.

User request:
{question}

Use clear paragraphs and headings when useful. Do not claim that information came from an uploaded document unless paper mode is explicitly selected."""

        if paper_mode and last_uploaded_file:
            contents = [prompt, last_uploaded_file]
        else:
            contents = prompt

        text, model = generate_text(contents, smart=False, attempts=1, max_output_tokens=1400)
        return {
            "success": True,
            "answer": text.strip(),
            "model": model,
            "mode": mode
        }
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": error_message(exc)}
        )


@app.post("/ask-stream")
async def ask_stream(request: AskRequest):
    """Low-latency streaming endpoint used by Ask BloomAI and voice questions."""
    question = request.question.strip()
    if not question:
        raise HTTPException(400, "Please enter or speak a question.")
    if len(question) > 3000:
        raise HTTPException(400, "Question is too long.")

    mode = (request.mode or "general").lower()
    paper_mode = mode.startswith("paper")
    answer_mode = mode.replace("paper_", "", 1) if paper_mode else mode

    # Exact answer-length targets so the Question Bank buttons produce
    # genuinely different, mark-appropriate answers.
    if answer_mode == "simple":
        mode_instruction = (
            "Explain ONLY this exact question in simple student-friendly language. "
            "Use about 60–100 words, focusing on the core definition, points and one "
            "small example only when it directly helps."
        )
    elif answer_mode == "5mark":
        mode_instruction = (
            "Write an exam-ready answer for EXACTLY this question, suitable for 5 marks. "
            "Target about 120–180 words. Include a brief introduction/definition followed "
            "by the key points, explanation and a relevant example where appropriate. "
            "Do not answer a broader topic and do not pad the response."
        )
    elif answer_mode == "10mark":
        mode_instruction = (
            "Write an exam-ready detailed answer for EXACTLY this question, suitable for "
            "10 marks. Target about 220–320 words. Cover the important sub-points, explain "
            "each clearly, and include examples/steps/advantages or comparisons only when "
            "they directly answer the question. Do not drift into unrelated material."
        )
    elif answer_mode == "short":
        mode_instruction = "Give a concise answer to EXACTLY this question, covering its essential points."
    elif answer_mode == "important":
        mode_instruction = "Answer EXACTLY this question and emphasize the most exam-relevant points."
    elif answer_mode == "paper":
        mode_instruction = (
            "Answer EXACTLY this question using the uploaded question paper as the primary "
            "academic source. Match its terminology, subject level and concepts."
        )
    else:
        mode_instruction = "Answer the actual request directly as a general-purpose academic assistant."

    metadata = []
    if request.marks is not None:
        metadata.append(f"Marks assigned to this question: {request.marks}")
    if request.bloom_level:
        metadata.append(f"Bloom level: {request.bloom_level}")
    if request.difficulty:
        metadata.append(f"Difficulty: {request.difficulty}")
    if request.topic:
        metadata.append(f"Topic: {request.topic}")
    metadata_text = "\n".join(metadata)

    paperContextInstruction = (
        "\nGROUNDING RULE: This is the exact question selected by the student from the "
        "Question Bank. Answer THAT question and nothing broader. Use the uploaded/analyzed "
        "paper as the primary source and stay within its terminology, topics and level. "
        "Do not replace the question with a generic answer to the topic. If the paper does "
        "not support a paper-specific detail, do not invent it; use only academically "
        "necessary explanation that directly answers the question."
        if paper_mode else ""
    )
    prompt = f"""You are BloomAI, a fast academic answer assistant.

{mode_instruction}

User's exact question:
{question}

Question metadata:
{metadata_text or "No additional metadata supplied."}
{paperContextInstruction}

Return ONLY the answer to the question. Use short paragraphs or numbered points when they improve exam readability. """
    contents = [prompt, last_uploaded_file] if paper_mode and last_uploaded_file else prompt

    def iterator():
        try:
            yield from stream_text(contents, max_output_tokens=1400)
        except Exception as exc:
            yield "__BLOOM_ERROR__" + error_message(exc)

    return StreamingResponse(iterator(), media_type="text/plain; charset=utf-8", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _compact_quiz_answer(value: Any, limit: int = 170) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text)
    text = parts[0] if parts and len(parts[0]) >= 25 else text
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def _quiz_relatedness(q1: dict, q2: dict) -> int:
    score = 0
    t1 = {str(x).strip().lower() for x in (q1.get("topics") or [])}
    t2 = {str(x).strip().lower() for x in (q2.get("topics") or [])}
    score += 8 * len(t1 & t2)
    if q1.get("bloom_level") and q1.get("bloom_level") == q2.get("bloom_level"):
        score += 2
    try:
        if int(q1.get("marks") or 0) == int(q2.get("marks") or 0):
            score += 2
    except Exception:
        pass
    return score


@app.post("/quiz")
async def quiz(request: QuizRequest):
    """Build a paper-grounded MCQ quiz.
    Every question and every option must come from the uploaded paper's analyzed
    questions/answers and share the same topic context where possible.
    No generic placeholders or unrelated-paper options are allowed.
    """
    if not last_analysis or not last_analysis.get("questions"):
        raise HTTPException(400, "Analyze a question paper first.")

    try:
        source = [
            q for q in (last_analysis.get("questions") or [])
            if q.get("question") and str(q.get("answer") or "").strip()
        ]
        if len(source) < 4:
            raise RuntimeError(
                "The uploaded paper does not contain enough analyzed questions and answers "
                "to build a reliable paper-grounded quiz."
            )

        def norm_words(value):
            words = re.findall(r"[a-zA-Z0-9]{3,}", str(value or "").lower())
            stop = {
                "the","and","for","with","from","that","this","what","which","where",
                "when","how","why","into","about","their","there","are","was","were",
                "have","has","been","will","would","could","should","does","using",
                "explain","write","brief","short","note","list","mention","define"
            }
            return {w for w in words if w not in stop}

        def context_tokens(q):
            topics = " ".join(str(x) for x in (q.get("topics") or []))
            return norm_words(topics) | norm_words(q.get("question"))

        def relatedness(q1, q2):
            a = context_tokens(q1)
            b = context_tokens(q2)
            overlap = len(a & b)
            topic1 = {str(x).strip().lower() for x in (q1.get("topics") or []) if str(x).strip()}
            topic2 = {str(x).strip().lower() for x in (q2.get("topics") or []) if str(x).strip()}
            exact_topic = len(topic1 & topic2)
            score = exact_topic * 100 + overlap * 8
            if q1.get("bloom_level") and q1.get("bloom_level") == q2.get("bloom_level"):
                score += 4
            try:
                if int(q1.get("marks") or 0) == int(q2.get("marks") or 0):
                    score += 3
            except Exception:
                pass
            return score, exact_topic, overlap

        count = max(3, min(int(request.count or 10), min(15, len(source))))
        # Deterministic rotation so repeated openings don't always begin with q1.
        offset = (len(source) * 7 + len(last_filename or "")) % len(source)
        rotated = source[offset:] + source[:offset]

        quiz_items = []
        for q in rotated:
            if len(quiz_items) >= count:
                break

            correct_full = str(q.get("answer") or "").strip()
            correct = _compact_quiz_answer(correct_full, 220)
            if not correct:
                continue

            candidates = []
            for other in source:
                if other is q:
                    continue
                answer = _compact_quiz_answer(other.get("answer"), 220)
                if not answer or answer == correct:
                    continue

                score, exact_topic, overlap = relatedness(q, other)

                # Strict grounding: require at least one shared paper topic.
                # If topic metadata is sparse, allow strong word overlap between
                # the actual question and another analyzed question.
                if exact_topic == 0 and overlap < 2:
                    continue

                candidates.append((score, exact_topic, overlap, answer, other))

            candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)

            distractors = []
            seen = {correct}
            for _, _, _, answer, other in candidates:
                if answer in seen:
                    continue
                distractors.append(answer)
                seen.add(answer)
                if len(distractors) == 3:
                    break

            # Never pad with unrelated content. A shorter quiz is preferable
            # to a quiz containing random options.
            if len(distractors) < 3:
                continue

            options = [correct] + distractors
            shift = (len(quiz_items) * 3 + len(str(q.get("question", "")))) % 4
            options = options[shift:] + options[:shift]

            quiz_items.append({
                "question": str(q.get("question", "")),
                "options": options,
                "answer_index": options.index(correct),
                "explanation": correct_full,
                "source_topics": q.get("topics") or [],
                "paper": q.get("paper") or "",
                "marks": q.get("marks") or 0,
                "bloom_level": q.get("bloom_level") or "",
                "difficulty": q.get("difficulty") or ""
            })

        if len(quiz_items) < 3:
            raise RuntimeError(
                "This paper does not contain enough same-topic analyzed answers to create "
                "three-option distractors safely. The quiz was not filled with unrelated questions."
            )

        return {
            "success": True,
            "quiz": quiz_items[:count],
            "model": "paper-grounded-local",
            "source": last_filename or "Uploaded Question Paper"
        }
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": error_message(exc)}
        )


class QuizPdfRequest(BaseModel):
    quiz: list[dict]
    score: int = 0
    total: int = 0
    source: str = ""


@app.post("/quiz-pdf")
async def quiz_pdf(request: QuizPdfRequest):
    """Download the submitted quiz with the questions and correct answers."""
    try:
        if not request.quiz:
            raise RuntimeError("Submit the quiz first.")

        buffer = __import__("io").BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=18 * mm,
            leftMargin=18 * mm,
            topMargin=18 * mm,
            bottomMargin=18 * mm,
            title="BloomAI Quiz",
            author="BloomAI",
        )
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "QuizTitle", parent=styles["Title"], fontName=REPORT_FONT_BOLD,
            fontSize=20, leading=24, alignment=TA_CENTER, spaceAfter=8
        )
        meta_style = ParagraphStyle(
            "QuizMeta", parent=styles["Normal"], fontName=REPORT_FONT,
            fontSize=9.5, leading=14, alignment=TA_CENTER,
            textColor=colors.HexColor("#5d6b76"), spaceAfter=16
        )
        q_style = ParagraphStyle(
            "QuizQ", parent=styles["BodyText"], fontName=REPORT_FONT_BOLD,
            fontSize=10.5, leading=15, spaceBefore=10, spaceAfter=6
        )
        option_style = ParagraphStyle(
            "QuizOption", parent=styles["BodyText"], fontName=REPORT_FONT,
            fontSize=9.2, leading=14, leftIndent=8, spaceAfter=4
        )
        answer_style = ParagraphStyle(
            "QuizAnswer", parent=styles["BodyText"], fontName=REPORT_FONT_BOLD,
            fontSize=9.5, leading=14, textColor=colors.HexColor("#176b4b"),
            spaceBefore=5, spaceAfter=10
        )

        story = [
            Paragraph("BloomAI Quiz", title_style),
            Paragraph(
                f"{pdf_escape(request.source or last_filename or 'Uploaded Question Paper')} "
                f"&nbsp; | &nbsp; Score: <b>{int(request.score or 0)}/{int(request.total or len(request.quiz))}</b>",
                meta_style,
            ),
        ]

        for i, item in enumerate(request.quiz, 1):
            question = pdf_escape(item.get("question", ""))
            story.append(Paragraph(f"{i}. {question}", q_style))
            options = item.get("options") or []
            correct_index = item.get("answer_index")
            for j, option in enumerate(options):
                letter = chr(65 + j)
                story.append(Paragraph(f"{letter}. {pdf_escape(option)}", option_style))

            if isinstance(correct_index, int) and 0 <= correct_index < len(options):
                correct_text = options[correct_index]
            else:
                correct_text = item.get("explanation") or ""
            story.append(
                Paragraph(
                    f"<b>Correct Answer:</b> {pdf_escape(correct_text)}",
                    answer_style,
                )
            )
            meta = []
            if item.get("marks") not in (None, "", 0):
                meta.append(f"Marks: {pdf_escape(item.get('marks'))}")
            if item.get("bloom_level"):
                meta.append(f"Bloom: {pdf_escape(item.get('bloom_level'))}")
            if item.get("difficulty"):
                meta.append(f"Difficulty: {pdf_escape(item.get('difficulty'))}")
            if item.get("source_topics"):
                meta.append(f"Topics: {pdf_escape(', '.join(map(str, item.get('source_topics') or [])))}")
            if meta:
                story.append(Paragraph(" &nbsp; | &nbsp; ".join(meta), meta_style))

        doc.build(story)
        data = buffer.getvalue()
        safe = re.sub(
            r"[^a-zA-Z0-9._-]",
            "_",
            Path(request.source or last_filename or "BloomAI_Quiz").stem
        )
        return Response(
            content=data,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="BloomAI_{safe}_Quiz.pdf"',
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "Pragma": "no-cache",
                "Content-Length": str(len(data)),
            },
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": error_message(exc)}
        )


class RecommendationRequest(BaseModel):
    count: int = 6
    difficulty: str = ""
    bloom_level: str = ""


def build_recommendations(count: int = 6, difficulty: str = "", bloom_level: str = ""):
    d = last_analysis or {}
    questions = d.get("questions") or []
    topics = d.get("topic_distribution") or []
    avg_diff = str(d.get("average_difficulty") or "Moderate")
    chosen_diff = difficulty if difficulty in {"Easy", "Moderate", "Difficult", "Very Difficult"} else avg_diff
    valid_bloom = {"Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"}
    chosen_bloom = bloom_level if bloom_level in valid_bloom else ""

    present_bloom = {str(q.get("bloom_level")) for q in questions if q.get("bloom_level")}
    topic_names = [str(t.get("topic") or "").strip() for t in topics if str(t.get("topic") or "").strip()]
    topic_names = topic_names[:12]
    if not topic_names:
        topic_names = ["the core concepts covered in the paper"]

    if not chosen_bloom:
        for level in ["Create", "Evaluate", "Analyze", "Apply", "Understand", "Remember"]:
            if level not in present_bloom:
                chosen_bloom = level
                break
        chosen_bloom = chosen_bloom or "Apply"

    stems = {
        "Remember": "Define {topic} and list the key points a student should remember.",
        "Understand": "Explain {topic} in your own words and illustrate it with a suitable example.",
        "Apply": "Given a practical situation involving {topic}, explain how you would apply the concept to solve it.",
        "Analyze": "Analyze {topic} by breaking it into its main components and explain how those components relate.",
        "Evaluate": "Evaluate the strengths, limitations and appropriate use of {topic} for a real-world scenario.",
        "Create": "Design a suitable solution, model or workflow using {topic} for a realistic academic or industry scenario."
    }
    reasons = {
        "Easy": "Adds a foundational question and strengthens coverage without raising the paper difficulty too much.",
        "Moderate": "Matches the paper's core level and adds balanced conceptual/application coverage.",
        "Difficult": "Raises the analytical depth while staying within the paper's existing subject scope.",
        "Very Difficult": "Adds a higher-order challenge suitable for distinguishing advanced preparation."
    }

    recs=[]
    for i in range(max(1, min(int(count or 6), 10))):
        topic=topic_names[i % len(topic_names)]
        # Rotate Bloom levels slightly when user did not explicitly select one.
        level=chosen_bloom
        if not bloom_level:
            cycle=["Apply","Analyze","Evaluate","Create","Understand","Remember"]
            level=cycle[i % len(cycle)]
        recs.append({
            "question": stems[level].format(topic=topic),
            "topic": topic,
            "difficulty": chosen_diff,
            "bloom_level": level,
            "suggested_marks": 2 if chosen_diff=="Easy" and level in {"Remember","Understand"} else (10 if level in {"Analyze","Evaluate","Create"} else 5),
            "reason": reasons[chosen_diff]
        })
    return recs


@app.post("/recommendations")
async def recommendations(request: RecommendationRequest):
    if not last_analysis or not last_analysis.get("questions"):
        raise HTTPException(400, "Analyze a question paper first.")
    return {"success": True, "recommendations": build_recommendations(request.count, request.difficulty, request.bloom_level)}


@app.post("/study-plan")
async def study_plan(request: StudyRequest):
    if not last_analysis:
        raise HTTPException(400, "Analyze a question paper first.")
    try:
        compact = {
            "subject": last_analysis.get("subject"),
            "papers": last_analysis.get("papers"),
            "topics": last_analysis.get("topic_distribution"),
            "difficulties": last_analysis.get("difficulty_distribution"),
            "bloom": last_analysis.get("bloom_distribution"),
            "priorities": last_analysis.get("study_priorities"),
            "important_questions": last_analysis.get("important_questions", [])[:20]
        }
        prompt = f'''Create a practical 3-day study plan using ONLY this question-paper analysis.
Return ONLY JSON with exactly these keys:
{{"summary":"","priorities":[""],"day_1":[""],"day_2":[""],"day_3":[""],"quick_revision":[""]}}
Keep each item concise and actionable. Prioritize repeated, difficult, high-mark and important topics found in the source.
SOURCE:
{json.dumps(compact, ensure_ascii=False)}'''
        text, model = generate_text(prompt, smart=False, attempts=1)
        plan = parse_json(text)
        return {"success": True, "plan": plan, "model": model}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"success": False, "message": error_message(exc)})


def pdf_escape(value: Any) -> str:
    """Escape text for ReportLab Paragraph XML."""
    import html
    return html.escape(str(value or "")).replace("\n", "<br/>")


REPORT_FONT = "Helvetica"
REPORT_FONT_BOLD = "Helvetica-Bold"
for _font_path, _font_name, _bold in [
    (r"C:\Windows\Fonts\arial.ttf", "BloomArial", False),
    (r"C:\Windows\Fonts\arialbd.ttf", "BloomArialBold", True),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "BloomDejaVu", False),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "BloomDejaVuBold", True),
]:
    try:
        if Path(_font_path).exists():
            pdfmetrics.registerFont(TTFont(_font_name, _font_path))
            if _bold:
                REPORT_FONT_BOLD = _font_name
            elif REPORT_FONT == "Helvetica":
                REPORT_FONT = _font_name
    except Exception:
        pass


def build_report_pdf() -> bytes:
    """Build a study-friendly PDF report from the latest analysis."""
    if not last_analysis:
        raise RuntimeError("Analyze a question paper first.")

    d = last_analysis
    buffer = __import__("io").BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="BloomAI Academic Analysis Report",
        author="BloomAI",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "BloomTitle",
        parent=styles["Title"],
        fontName=REPORT_FONT_BOLD,
        fontSize=22,
        leading=27,
        alignment=TA_CENTER,
        spaceAfter=12,
    )
    subtitle_style = ParagraphStyle(
        "BloomSubtitle",
        parent=styles["Normal"],
        fontName=REPORT_FONT,
        fontSize=9.5,
        leading=14,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#555555"),
        spaceAfter=18,
    )
    heading_style = ParagraphStyle(
        "BloomHeading",
        parent=styles["Heading2"],
        fontName=REPORT_FONT_BOLD,
        fontSize=14,
        leading=18,
        spaceBefore=15,
        spaceAfter=8,
        textColor=colors.HexColor("#0f766e"),
    )
    body_style = ParagraphStyle(
        "BloomBody",
        parent=styles["BodyText"],
        fontName=REPORT_FONT,
        fontSize=9.5,
        leading=15,
        spaceAfter=8,
    )
    question_style = ParagraphStyle(
        "BloomQuestion",
        parent=body_style,
        fontName=REPORT_FONT_BOLD,
        fontSize=10,
        leading=15,
        spaceAfter=5,
    )
    meta_style = ParagraphStyle(
        "BloomMeta",
        parent=body_style,
        fontSize=8.5,
        textColor=colors.HexColor("#666666"),
        leading=13,
    )

    story = []
    story.append(Paragraph("BloomAI Academic Analysis Report", title_style))
    story.append(Paragraph(
        f"{pdf_escape(last_filename or 'Uploaded Question Paper')} &nbsp; | &nbsp; "
        f"{pdf_escape(d.get('subject', 'Academic Question Paper'))}",
        subtitle_style,
    ))

    story.append(Paragraph("Overview", heading_style))
    overview = (
        f"BloomAI analyzed <b>{int(d.get('total_questions', 0) or 0)}</b> questions "
        f"across <b>{int(d.get('paper_count', 1) or 1)}</b> paper(s). "
        f"The combined maximum marks identified were <b>{int(d.get('total_marks', 0) or 0)}</b>. "
        f"The overall average difficulty was <b>{pdf_escape(d.get('average_difficulty', 'Moderate'))}</b> "
        f"and the paper-quality indicator was <b>{int(d.get('paper_quality_score', 0) or 0)}/100</b>."
    )
    story.append(Paragraph(overview, body_style))

    story.append(Paragraph("Paper Details", heading_style))
    papers = d.get("papers") or []
    if papers:
        for pinfo in papers:
            story.append(Paragraph(
                f"<b>{pdf_escape(pinfo.get('label', 'Paper'))}</b> &nbsp; | &nbsp; "
                f"Year: {pdf_escape(pinfo.get('year', ''))} &nbsp; | &nbsp; "
                f"Questions: {int(pinfo.get('question_count', 0) or 0)} &nbsp; | &nbsp; "
                f"Maximum Marks: {int(pinfo.get('max_marks', 0) or 0)}",
                body_style,
            ))
    else:
        story.append(Paragraph("No separate paper metadata was returned.", body_style))

    story.append(Paragraph("Key Findings", heading_style))
    insights = d.get("insights") or []
    issues = d.get("issues") or []
    if insights:
        story.append(Paragraph(" ".join(pdf_escape(x) for x in insights), body_style))
    else:
        story.append(Paragraph("No additional AI insights were returned.", body_style))
    if issues:
        story.append(Paragraph(
            "Potential issues identified: " + " ".join(pdf_escape(x) for x in issues),
            body_style,
        ))
    else:
        story.append(Paragraph("No major issues were identified from the supplied document.", body_style))

    story.append(Paragraph("Bloom's Taxonomy", heading_style))
    bloom = d.get("bloom_distribution") or {}
    bloom_sentence = ", ".join(
        f"{pdf_escape(k)}: {int(v or 0)} question(s)" for k, v in bloom.items()
    )
    story.append(Paragraph(bloom_sentence or "No Bloom data available.", body_style))

    story.append(Paragraph("Difficulty Distribution", heading_style))
    diff = d.get("difficulty_distribution") or {}
    diff_sentence = ", ".join(
        f"{pdf_escape(k)}: {int(v or 0)} question(s)" for k, v in diff.items()
    )
    story.append(Paragraph(diff_sentence or "No difficulty data available.", body_style))

    story.append(Paragraph("Topic Coverage", heading_style))
    topics = d.get("topic_distribution") or []
    if topics:
        story.append(Paragraph(
            ", ".join(
                f"{pdf_escape(x.get('topic', ''))} ({int(x.get('count', 0) or 0)})"
                for x in topics[:15]
            ),
            body_style,
        ))
    else:
        story.append(Paragraph("No topic distribution was returned.", body_style))

    story.append(Paragraph("Important Questions and Answers", heading_style))
    important = d.get("important_questions") or []
    if not important:
        important = sorted(
            d.get("questions") or [],
            key=lambda x: float(x.get("marks", 0) or 0),
            reverse=True,
        )[:10]

    for i, q in enumerate(important, 1):
        story.append(Paragraph(
            f"{i}. {pdf_escape(q.get('question', 'Question unavailable'))}",
            question_style,
        ))
        meta = (
            f"Marks: {q.get('marks', 0)} &nbsp; | &nbsp; "
            f"Bloom: {pdf_escape(q.get('bloom_level', ''))} &nbsp; | &nbsp; "
            f"Difficulty: {pdf_escape(q.get('difficulty', ''))}"
        )
        story.append(Paragraph(meta, meta_style))
        story.append(Paragraph(
            pdf_escape(q.get("answer") or "Answer could not be generated from the uploaded paper. Please use Ask BloomAI for this question."),
            body_style,
        ))

    story.append(Paragraph("Revision Priorities", heading_style))
    priorities = d.get("study_priorities") or []
    if priorities:
        for item in priorities:
            story.append(Paragraph(f"• {pdf_escape(item)}", body_style))
    else:
        story.append(Paragraph("Review difficult, repeated and high-mark questions first.", body_style))

    story.append(PageBreak())
    story.append(Paragraph("Question-wise Analysis", heading_style))

    for i, q in enumerate(d.get("questions") or [], 1):
        story.append(Paragraph(
            f"{i}. {pdf_escape(q.get('question', 'Question unavailable'))}",
            question_style,
        ))
        meta = (
            f"Marks: {q.get('marks', 0)} &nbsp; | &nbsp; "
            f"Paper: {pdf_escape(q.get('paper', ''))} &nbsp; | &nbsp; "
            f"Bloom: {pdf_escape(q.get('bloom_level', ''))} &nbsp; | &nbsp; "
            f"Difficulty: {pdf_escape(q.get('difficulty', ''))} "
            f"({q.get('difficulty_score', 0)}/100)"
        )
        story.append(Paragraph(meta, meta_style))
        topics_text = q.get("topics") or []
        if topics_text:
            story.append(Paragraph(
                "Topics: " + ", ".join(pdf_escape(x) for x in topics_text),
                meta_style,
            ))
        story.append(Paragraph(
            "Answer: " + pdf_escape(q.get("answer") or "Answer could not be generated from the uploaded paper. Please use Ask BloomAI for this question."),
            body_style,
        ))

    doc.build(story)
    return buffer.getvalue()


@app.get("/report-pdf")
async def report_pdf():
    try:
        pdf_bytes = build_report_pdf()
        safe = re.sub(r"[^a-zA-Z0-9._-]", "_", Path(last_filename or "BloomAI_Report").stem)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="BloomAI_{safe}_Report.pdf"', "Cache-Control": "no-store, no-cache, must-revalidate", "Pragma": "no-cache", "Content-Length": str(len(pdf_bytes))},
        )
    except Exception as exc:
        return JSONResponse(status_code=500, content={"success": False, "message": error_message(exc)})


HTML_PAGE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BloomAI — Academic Intelligence</title>
<style>
:root{--bg:#06131a;--bg2:#0a2028;--card:rgba(12,34,43,.88);--card2:#102b35;--line:rgba(45,212,191,.18);--text:#f2fbfa;--muted:#91a8ad;--p:#14b8a6;--p2:#2dd4bf;--b:#0ea5e9;--g:#22c55e;--r:#fb7185;--o:#f59e0b;--shadow:0 25px 80px rgba(0,0,0,.38)}
body.light{--bg:#eef8f7;--bg2:#ffffff;--card:rgba(255,255,255,.94);--card2:#ffffff;--line:rgba(13,148,136,.18);--text:#102a2d;--muted:#607579;--shadow:0 20px 60px rgba(20,90,90,.12)}
*{box-sizing:border-box}html{scroll-behavior:smooth}select{background:var(--card2)!important;color:var(--text)!important;border:1px solid var(--line)!important;color-scheme:dark}select option{background:var(--card2);color:var(--text)}body.light select{color-scheme:light}body.light select option{background:#fff;color:#172432}body{margin:0;min-height:100vh;color:var(--text);font-family:Inter,Segoe UI,system-ui,sans-serif;background:radial-gradient(circle at 12% 8%,rgba(20,184,166,.20),transparent 28%),radial-gradient(circle at 90% 14%,rgba(14,165,233,.12),transparent 26%),radial-gradient(circle at 55% 85%,rgba(14,165,233,.10),transparent 30%),linear-gradient(135deg,var(--bg),var(--bg2));background-attachment:fixed;transition:.25s;overflow-x:hidden}body:before{content:"";position:fixed;inset:-20%;z-index:-1;pointer-events:none;background:radial-gradient(circle at 30% 30%,rgba(45,212,191,.10),transparent 24%),radial-gradient(circle at 70% 60%,rgba(14,165,233,.07),transparent 22%);filter:blur(30px);animation:aurora 18s ease-in-out infinite alternate}@keyframes aurora{0%{transform:translate3d(-2%,0,0) scale(1)}50%{transform:translate3d(2%,-2%,0) scale(1.04)}100%{transform:translate3d(-1%,2%,0) scale(1.02)}}button,input,textarea,select{font:inherit}button{cursor:pointer}.wrap{width:min(1220px,calc(100% - 34px));margin:auto}
.nav{position:sticky;top:0;z-index:80;height:72px;backdrop-filter:blur(20px);background:rgba(6,19,26,.78);border-bottom:1px solid var(--line)}body.light .nav{background:rgba(255,255,255,.76)}.navin{height:100%;display:flex;align-items:center;justify-content:space-between}.brand{display:flex;align-items:center;gap:12px;font-weight:850;font-size:21px}.logo{width:40px;height:40px;border-radius:13px;display:grid;place-items:center;color:#fff;background:linear-gradient(135deg,var(--p),var(--b));box-shadow:0 10px 30px rgba(139,92,246,.35)}.navright{display:flex;gap:9px}.navbtn{border:1px solid var(--line);background:var(--card);color:var(--text);padding:9px 13px;border-radius:12px;font-size:13px}.navbtn:hover,.smallbtn:hover,.tool:hover{border-color:rgba(139,92,246,.55);transform:translateY(-1px)}
.layout{display:grid;grid-template-columns:235px 1fr;min-height:calc(100vh - 72px)}.side{position:sticky;top:72px;height:calc(100vh - 72px);padding:20px 13px;border-right:1px solid var(--line);background:rgba(6,19,26,.42);overflow:auto}.side h4{font-size:11px;color:var(--muted);letter-spacing:.12em;text-transform:uppercase;margin:14px 9px 8px}.sideitem{display:flex;align-items:center;gap:10px;width:100%;padding:11px 12px;border:1px solid transparent;border-radius:12px;color:var(--muted);background:transparent;text-align:left;margin:3px 0}.sideitem:hover,.sideitem.active{color:var(--text);background:rgba(20,184,166,.09);border-color:var(--line)}.sideitem.active{box-shadow:inset 3px 0 0 var(--p)}.main{min-width:0}.hero{padding:65px 0 30px;text-align:center}.badge{display:inline-flex;gap:8px;align-items:center;padding:8px 13px;border:1px solid var(--line);border-radius:999px;background:rgba(20,184,166,.09);font-size:12px;font-weight:750;color:#b7fff6}.dot{width:7px;height:7px;border-radius:50%;background:var(--g);box-shadow:0 0 14px var(--g)}.hero h1{font-size:clamp(42px,6vw,72px);letter-spacing:-3px;line-height:.98;margin:18px 0;background:linear-gradient(135deg,#fff,#99f6e4,#38bdf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent}.hero p{max-width:700px;margin:auto;color:var(--muted);line-height:1.7}
.upload{max-width:900px;margin:20px auto 42px;padding:10px;border:1px solid var(--line);border-radius:28px;background:linear-gradient(145deg,rgba(20,184,166,.10),rgba(99,102,241,.03));box-shadow:var(--shadow)}.uploadin{padding:45px 25px;border:1px dashed rgba(167,139,250,.38);border-radius:21px;text-align:center}.uploadicon{width:70px;height:70px;border-radius:21px;margin:auto auto 18px;display:grid;place-items:center;font-size:30px;background:rgba(139,92,246,.13)}.upload h2{margin:0 0 8px}.upload p{color:var(--muted)}input[type=file]{display:none}.filelabel,.primary{display:inline-flex;align-items:center;justify-content:center;gap:8px;border:0;border-radius:13px;padding:13px 19px;color:#fff;background:linear-gradient(135deg,var(--p),var(--b));font-weight:800;box-shadow:0 12px 30px rgba(14,165,233,.25)}.filename{min-height:20px;margin-top:15px;color:var(--p2);font-size:13px}.primary:disabled{opacity:.5;cursor:not-allowed}.progress{display:none;max-width:900px;margin:0 auto 35px}.progressbox{padding:18px 20px;border:1px solid var(--line);border-radius:18px;background:var(--card)}.progresshead{display:flex;justify-content:space-between;margin-bottom:9px}.track{height:8px;border-radius:99px;background:rgba(139,92,246,.08);overflow:hidden}.fill{height:100%;width:0;background:linear-gradient(90deg,var(--p),var(--b));transition:width .35s}.muted{color:var(--muted)}.error{display:none;max-width:900px;margin:0 auto 25px;padding:15px;border:1px solid rgba(251,113,133,.25);background:rgba(251,113,133,.07);border-radius:13px;color:#fecdd3}
.ask{max-width:900px;margin:0 auto 50px}.sectiontitle{display:flex;justify-content:space-between;align-items:end;margin-bottom:14px}.sectiontitle h2{margin:0;font-size:22px}.askmode{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:13px}.askmode label{font-size:12px;color:var(--muted);font-weight:700}.askmode select{padding:9px 12px;border:1px solid var(--line);border-radius:11px;background:var(--card2);color:var(--text);outline:0}.askcard{padding:22px;border:1px solid var(--line);border-radius:22px;background:var(--card);box-shadow:var(--shadow)}.askrow{display:flex;gap:10px}.asktext{flex:1;min-height:62px;resize:vertical;border:1px solid var(--line);outline:0;border-radius:15px;padding:16px;color:var(--text);background:rgba(0,0,0,.12)}body.light .asktext{background:rgba(99,102,241,.04)}.asktext:focus{border-color:rgba(139,92,246,.55)}.mic{width:58px;border:1px solid var(--line);border-radius:15px;background:rgba(139,92,246,.14);color:var(--text);font-size:21px}.mic.listen{animation:pulse 1.1s infinite;background:rgba(251,113,133,.18)}@keyframes pulse{50%{box-shadow:0 0 0 12px rgba(251,113,133,0)}}.askbtn{height:58px;padding:0 20px;border:0;border-radius:15px;background:linear-gradient(135deg,var(--p),var(--b));color:#fff;font-weight:800}.status{min-height:22px;margin-top:10px;font-size:13px;color:var(--muted)}.thinking:after{content:'...';display:inline-block;width:0;overflow:hidden;vertical-align:bottom;animation:dots 1.2s steps(4,end) infinite}@keyframes dots{to{width:1.3em}}.answer{display:none;margin-top:15px;padding:19px;border:1px solid var(--line);border-radius:17px;background:rgba(20,184,166,.055);line-height:1.75;white-space:pre-wrap}.answeractions{display:none;gap:8px;margin-top:12px;flex-wrap:wrap}.tool,.smallbtn{border:1px solid var(--line);background:rgba(139,92,246,.07);color:var(--text);padding:9px 12px;border-radius:11px;font-size:12px}.smallbtn{display:inline-flex;align-items:center;gap:6px}
.dash{display:none;padding:25px 0 70px}.dashhead{display:flex;justify-content:space-between;align-items:flex-end;gap:15px;margin-bottom:20px}.dashhead h2{margin:0 0 5px;font-size:32px}.dashactions{display:flex;gap:8px;flex-wrap:wrap}.stats{display:grid;grid-template-columns:repeat(5,1fr);gap:13px;margin-bottom:18px}.stat{padding:20px;border:1px solid var(--line);border-radius:18px;background:var(--card)}.stat small{color:var(--muted)}.stat b{display:block;font-size:27px;margin-top:8px}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:17px;margin-bottom:17px}.panel{padding:21px;border:1px solid var(--line);border-radius:19px;background:var(--card);transition:transform .25s ease,box-shadow .25s ease}.panel:hover{transform:translateY(-2px);box-shadow:0 18px 45px rgba(99,102,241,.08)}.upload,.askcard,.stat,.question{transition:transform .25s ease,box-shadow .25s ease}.upload:hover,.askcard:hover,.stat:hover{box-shadow:0 20px 60px rgba(14,165,233,.10)}.panel h3{margin:0 0 17px}.barrow{margin:0 0 13px}.barinfo{display:flex;justify-content:space-between;font-size:12px;margin-bottom:6px}.barinfo span{color:var(--muted)}.bartrack{height:8px;background:rgba(139,92,246,.08);border-radius:99px;overflow:hidden}.barfill{height:100%;border-radius:99px;background:linear-gradient(90deg,var(--p),var(--b));transition:width .7s}.chips{display:flex;flex-wrap:wrap;gap:8px}.chip{padding:8px 10px;border:1px solid var(--line);border-radius:999px;background:rgba(139,92,246,.06);font-size:12px}.papers,.list{display:grid;gap:9px}.item{padding:12px;border:1px solid var(--line);border-radius:12px;background:rgba(139,92,246,.045);line-height:1.55}.topicgrid{display:grid;grid-template-columns:1fr 1fr;gap:17px;margin-bottom:17px}.searchrow{display:flex;gap:10px;margin-bottom:13px}.search{flex:1;padding:13px;border-radius:13px;border:1px solid var(--line);outline:0;color:var(--text);background:rgba(0,0,0,.12)}body.light .search{background:rgba(99,102,241,.04)}.filters{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:15px}.filter{padding:7px 10px;border-radius:999px;border:1px solid var(--line);background:transparent;color:var(--muted);font-size:11px}.filter.active{color:#fff;background:linear-gradient(135deg,var(--p),var(--b));border-color:transparent}.questions{display:grid;gap:9px}.question{border:1px solid var(--line);border-radius:16px;overflow:hidden;background:rgba(20,184,166,.035)}.qtop{display:flex;align-items:center;gap:10px;padding:15px;cursor:pointer}.qnum{width:35px;height:35px;border-radius:10px;display:grid;place-items:center;color:#fff;background:linear-gradient(135deg,var(--p),var(--b));font-weight:800;font-size:12px;flex-shrink:0}.qtext{flex:1;font-weight:650;line-height:1.45}.tag{padding:5px 8px;border-radius:999px;background:rgba(20,184,166,.10);color:var(--p2);font-size:10px;font-weight:750}.qbody{display:none;padding:0 16px 12px 60px;color:var(--muted);white-space:pre-wrap;line-height:1.62}.question.open .qbody{display:block}.qtools{display:flex;gap:8px;flex-wrap:wrap;margin-top:9px;padding-bottom:2px}.empty{padding:18px;text-align:center;color:var(--muted)}
.modal{display:none;position:fixed;inset:0;z-index:200;background:rgba(0,0,0,.7);backdrop-filter:blur(8px);padding:40px 15px;overflow:auto}.modal.show{display:block}.modalbox{width:min(900px,100%);margin:auto;border:1px solid var(--line);border-radius:22px;background:var(--card2);box-shadow:var(--shadow);padding:23px}.modalhead{display:flex;justify-content:space-between;align-items:center;margin-bottom:18px}.modalbody{line-height:1.7}.quizq{padding:17px;border:1px solid var(--line);border-radius:15px;margin-bottom:11px}.option{display:block;padding:9px;margin-top:7px;border-radius:10px;background:rgba(20,184,166,.05)}.score{font-size:24px;font-weight:850;margin-top:16px}.report{display:none}.report h1{font-size:34px}.report p{line-height:1.8}.report h2{margin-top:28px}.report .reportq{padding:14px 0;border-bottom:1px solid var(--line)}
.reporttoolbar{display:flex;gap:9px;flex-wrap:wrap;margin-bottom:16px;padding:12px;border:1px solid var(--line);border-radius:15px;background:rgba(20,184,166,.055)}
.reportpreview{max-height:72vh;overflow:auto;padding:30px;background:#fff;color:#171717;border-radius:16px;box-shadow:inset 0 0 0 1px #e7e7ef}
.reportpreview h1{font-size:28px;color:#24213a;margin:0 0 8px}.reportpreview h2{font-size:18px;color:#0f766e;margin:26px 0 9px}.reportpreview p{color:#27272a;line-height:1.75;margin:7px 0}.reportpreview .rpq{padding:14px 0;border-bottom:1px solid #e5e7eb}.reportpreview .rpmeta{font-size:12px;color:#6b7280;margin:4px 0 8px}.reportpreview .rpanswer{white-space:pre-wrap}.report-help{display:none;margin-top:10px;padding:12px 14px;border-radius:12px;border:1px solid rgba(251,113,133,.25);background:rgba(251,113,133,.07);color:var(--muted);line-height:1.55;font-size:12px}.report-help.show{display:block}.report-help b{color:var(--text)}
footer{text-align:center;padding:35px;color:var(--muted);font-size:12px}
@media(max-width:1050px){.layout{grid-template-columns:190px 1fr}.stats{grid-template-columns:repeat(3,1fr)}}@media(max-width:800px){.layout{display:block}.side{position:fixed;left:-260px;top:72px;width:245px;z-index:100;transition:.25s;background:var(--bg2)}.side.open{left:0}.mobilemenu{display:inline-flex}.grid2,.topicgrid{grid-template-columns:1fr}.stats{grid-template-columns:repeat(2,1fr)}.dashhead{align-items:flex-start;flex-direction:column}}@media(min-width:801px){.mobilemenu{display:none}}@media(max-width:600px){.nav{height:64px}.navin{padding:0 10px}.brand{font-size:18px}.logo{width:36px;height:36px;border-radius:11px}.navbtn{padding:8px 10px}.navright{gap:6px}.navright .navbtn:nth-last-child(2){display:none}.wrap{width:calc(100% - 22px)}.hero{padding-top:45px}.hero h1{letter-spacing:-2px}.askrow{flex-wrap:wrap}.asktext{flex-basis:100%}.askbtn{flex:1}.qtop{align-items:flex-start}.tag{display:none}.qbody{padding-left:15px}}
@media print{body{background:#fff!important;color:#111!important}.nav,.side,.hero,.upload,.ask,.dashhead,.dashactions,.progress,.error,footer,#dashboard,.modal,.reporttoolbar{display:none!important}.report{display:block!important;width:100%;color:#111!important}.report p,.reportq{color:#111!important}.report h1,.report h2{color:#111!important}.reportpreview{max-height:none;overflow:visible;box-shadow:none;border:0;padding:0}.reportpreview h2{color:#111!important}}

/* BloomAI Midnight Ocean theme motion */
@keyframes bloomPulse{0%,100%{box-shadow:0 0 0 0 rgba(45,212,191,0)}50%{box-shadow:0 0 0 8px rgba(45,212,191,.035)}}
@keyframes floatIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
.logo{animation:bloomPulse 3.5s ease-in-out infinite}
.hero .badge{animation:floatIn .7s ease both}
.hero h1{animation:floatIn .85s .05s ease both}
.hero p{animation:floatIn .95s .1s ease both}
.primary{background-size:200% 100%;background-image:linear-gradient(110deg,var(--p),var(--b),var(--p));transition:transform .2s ease,box-shadow .2s ease,background-position .5s ease}
.primary:hover:not(:disabled){transform:translateY(-2px);background-position:100% 0;box-shadow:0 15px 34px rgba(20,184,166,.25)}
.sideitem:hover,.navbtn:hover,.smallbtn:hover,.tool:hover{transform:translateX(2px);transition:.2s ease}
@media(max-width:800px){
  .modal{padding:18px 10px}
  .modalbox{padding:16px;border-radius:18px}
  .reporttoolbar{position:sticky;top:0;z-index:2}
  .reportpreview{padding:18px}
}

/* BloomAI — Reference Dashboard Theme
   Inspired by the supplied Bloom's Dashboard screenshots.
   Dark: charcoal/navy + slate cards + electric blue accents.
   Light: clean white/slate + the same blue/green/orange/pink accents.
*/
:root{
 --bg:#090e14;
 --bg2:#0d141c;
 --card:#151d27;
 --card2:#18222d;
 --line:#2a3745;
 --text:#f2f6fa;
 --muted:#91a0af;
 --p:#2ea8ff;
 --p2:#73c7ff;
 --b:#168ee0;
 --g:#39d98a;
 --r:#ff5b72;
 --o:#f6ad24;
 --shadow:0 24px 70px rgba(0,0,0,.36);
 --grid:rgba(92,116,139,.08)
}
body.light{
 --bg:#f2f6fa;
 --bg2:#f8fafc;
 --card:#ffffff;
 --card2:#f7f9fc;
 --line:#d7e0e8;
 --text:#172432;
 --muted:#617283;
 --p:#168fe0;
 --p2:#287fba;
 --b:#1978bb;
 --g:#168b5a;
 --r:#d8425b;
 --o:#b87500;
 --shadow:0 20px 55px rgba(37,67,95,.12);
 --grid:rgba(71,99,126,.065)
}

body{
 background:
   linear-gradient(var(--grid) 1px,transparent 1px),
   linear-gradient(90deg,var(--grid) 1px,transparent 1px),
   radial-gradient(circle at 78% 4%,rgba(46,168,255,.10),transparent 28%),
   radial-gradient(circle at 8% 82%,rgba(57,217,138,.045),transparent 25%),
   var(--bg);
 background-size:40px 40px,40px 40px,auto,auto,auto;
 color:var(--text);
}
body:before{
 background:
   radial-gradient(circle at 24% 20%,rgba(46,168,255,.07),transparent 23%),
   radial-gradient(circle at 75% 72%,rgba(57,217,138,.035),transparent 20%);
 filter:blur(36px);
 opacity:.9;
}
body.light{
 background:
   linear-gradient(var(--grid) 1px,transparent 1px),
   linear-gradient(90deg,var(--grid) 1px,transparent 1px),
   radial-gradient(circle at 80% 4%,rgba(46,168,255,.09),transparent 28%),
   linear-gradient(180deg,#f8fafc,#f0f5f9);
 background-size:40px 40px,40px 40px,auto,auto;
}
.nav{
 background:rgba(9,14,20,.94);
 border-bottom:1px solid var(--line);
 box-shadow:0 8px 28px rgba(0,0,0,.14);
}
body.light .nav{
 background:rgba(248,250,252,.94);
 box-shadow:0 8px 28px rgba(37,67,95,.08);
}
.logo{
 background:linear-gradient(145deg,#42b7ff,#167ec0);
 box-shadow:0 9px 26px rgba(46,168,255,.24);
}
.navbtn,.smallbtn,.tool{
 background:rgba(255,255,255,.025);
 border-color:var(--line);
}
body.light .navbtn,
body.light .smallbtn,
body.light .tool{
 background:#fff;
}
.navbtn:hover,.smallbtn:hover,.tool:hover{
 border-color:rgba(46,168,255,.55);
 box-shadow:0 7px 20px rgba(46,168,255,.08);
}
.side{
 background:rgba(9,14,20,.78);
 border-right:1px solid var(--line);
}
body.light .side{
 background:rgba(248,250,252,.88);
}
.sideitem{
 border-color:transparent;
}
.sideitem:hover,.sideitem.active{
 background:rgba(46,168,255,.085);
 border-color:rgba(46,168,255,.20);
}
.sideitem.active{
 box-shadow:inset 3px 0 0 #2ea8ff;
}
.hero h1{
 background:linear-gradient(135deg,#ffffff,#c8eaff,#55b9ff);
 -webkit-background-clip:text;
 background-clip:text;
}
body.light .hero h1{
 background:linear-gradient(135deg,#17334d,#216f9e,#4d91bd);
 -webkit-background-clip:text;
 background-clip:text;
}
.badge{
 background:rgba(46,168,255,.075);
 border-color:rgba(46,168,255,.20);
 color:#a9ddff;
}
body.light .badge{
 color:#27688f;
 background:rgba(46,168,255,.07);
}
.dot{
 background:var(--g);
 box-shadow:0 0 14px rgba(57,217,138,.65);
}
.upload{
 background:linear-gradient(145deg,rgba(46,168,255,.075),rgba(255,255,255,.018));
 border-color:var(--line);
}
body.light .upload{
 background:linear-gradient(145deg,#ffffff,#f5f9fc);
}
.uploadin{
 border-color:rgba(46,168,255,.32);
}
.uploadicon{
 background:rgba(46,168,255,.10);
}
.filelabel,.primary,.askbtn{
 background:linear-gradient(135deg,#299fe7,#237db7);
 box-shadow:0 12px 28px rgba(31,139,211,.20);
}
.filelabel:hover,.primary:hover:not(:disabled),.askbtn:hover{
 box-shadow:0 16px 34px rgba(31,139,211,.27);
}
.progressbox,.panel,.stat,.askcard{
 backdrop-filter:blur(14px);
}
.panel:hover,.stat:hover,.askcard:hover,.upload:hover{
 box-shadow:0 20px 55px rgba(46,168,255,.08);
}
.barfill,.fill,.qnum{
 background:linear-gradient(90deg,#3ab2ff,#2488d0);
}
.track,.bartrack{
 background:rgba(255,255,255,.055);
}
body.light .track,body.light .bartrack{
 background:#e4ebf2;
}
.chip,.tool,.smallbtn{
 background:rgba(46,168,255,.045);
}
body.light .chip,body.light .tool,body.light .smallbtn{
 background:#f5f9fc;
}
.question{
 background:rgba(46,168,255,.022);
}
body.light .question{
 background:#fbfdff;
}
.tag{
 background:rgba(46,168,255,.085);
 color:#8fd4ff;
}
body.light .tag{
 color:#2b78a8;
 background:rgba(46,168,255,.075);
}
.asktext,.search{
 background:rgba(255,255,255,.025);
 border-color:var(--line);
}
body.light .asktext,body.light .search{
 background:#f8fafc;
}
.modalbox{
 background:linear-gradient(145deg,var(--card2),var(--card));
 box-shadow:0 30px 90px rgba(0,0,0,.48);
}
body.light .modalbox{
 box-shadow:0 30px 90px rgba(37,67,95,.16);
}
.option{
 background:rgba(46,168,255,.045);
 border:1px solid transparent;
}
.option:hover{
 background:rgba(46,168,255,.11);
 border-color:rgba(46,168,255,.22);
}
body.light .option{
 background:#f3f8fc;
}
body.light .option:hover{
 background:#eaf5fc;
 border-color:rgba(46,168,255,.24);
}
.recommend-card{
 background:linear-gradient(145deg,rgba(46,168,255,.055),rgba(255,255,255,.012));
}
body.light .recommend-card{
 background:linear-gradient(145deg,#ffffff,#f5f9fc);
}
.recommend-card{border:1px solid var(--line)!important;border-radius:16px!important;padding:16px!important;box-shadow:0 10px 28px rgba(0,0,0,.10);margin-bottom:10px}
.recommend-meta{display:flex;gap:7px;flex-wrap:wrap;margin:7px 0 7px}.recommend-reason{line-height:1.45}
.recommend-meta span{padding:4px 8px;border:1px solid var(--line);border-radius:999px;background:rgba(46,168,255,.07);font-size:10px;color:var(--muted)}
.filter.active{
 background:linear-gradient(135deg,#299fe7,#237db7);
}
.reporttoolbar{
 background:rgba(46,168,255,.045);
}
body.light .reporttoolbar{
 background:#f5f9fc;
}

/* Reference-style status accents */
.bloom-remember{color:#ff5b72!important}
.bloom-understand{color:#f6ad24!important}
.bloom-apply{color:#f6ad24!important}
.bloom-analyze{color:#2ea8ff!important}
.bloom-evaluate{color:#6f7cff!important}
.bloom-create{color:#39d98a!important}

/* Gentle motion, kept lightweight for mobile devices */
.panel,.stat,.question,.recommend-card,.upload,.askcard{
 animation:referenceRise .38s ease both;
}
@keyframes referenceRise{
 from{opacity:.01;transform:translateY(5px)}
 to{opacity:1;transform:translateY(0)}
}
@media(prefers-reduced-motion:reduce){
 *,*:before,*:after{animation:none!important;transition:none!important;scroll-behavior:auto!important}
}
@media(max-width:800px){
 .side{background:var(--bg2);box-shadow:12px 0 35px rgba(0,0,0,.18)}
 .panel,.stat,.question,.recommend-card{animation:none}
}


.quizq{margin:0 0 14px;padding:18px;border:1px solid var(--line);border-radius:16px;background:rgba(255,255,255,.018)}
.quizq .option{display:block;margin-top:8px;padding:12px 14px;border-radius:11px;cursor:pointer}
.quizq .option input{margin-right:7px}
.score{margin-top:14px;padding:14px 0}
</style>
</head>
<body>
<nav class="nav"><div class="wrap navin"><div class="brand"><div class="logo">✦</div>BloomAI</div><div class="navright"><button class="navbtn mobilemenu" onclick="toggleSide()">☰</button><button class="navbtn" onclick="scrollToId('ask')">Ask AI</button><button class="navbtn" onclick="scrollToId('dashboard')">Dashboard</button><button class="navbtn" onclick="toggleTheme()">☀</button></div></div></nav>
<div class="layout">
<aside class="side" id="side">
<h4>Workspace</h4>
<button class="sideitem active" onclick="navTo('dashboard')">📊 Overview</button>
<button class="sideitem" onclick="navTo('questionsPanel')">📝 Question Bank</button>
<button class="sideitem" onclick="navTo('recommendationsPanel')">💡 Recommendations</button>
<button class="sideitem" onclick="navTo('bloomPanel')">🧠 Bloom Analysis</button>
<button class="sideitem" onclick="navTo('difficultyPanel')">📈 Difficulty</button>
<button class="sideitem" onclick="navTo('topicPanel')">📚 Topics</button>
<button class="sideitem" onclick="navTo('importantPanel')">⭐ Important Questions</button>
<h4>Study Tools</h4>
<button class="sideitem" onclick="openQuiz()">🧪 Quiz Mode</button>
<button class="sideitem" onclick="openStudy()">📅 Study Plan</button>
<button class="sideitem" onclick="scrollToId('ask')">🎙️ Ask / Voice</button>
<button class="sideitem" onclick="downloadReportPDF()">📄 Download PDF Report</button>
<button class="sideitem" onclick="exportJSON()">↓ Export JSON</button>
<h4>Quick Actions</h4>
<button class="sideitem" onclick="newAnalysis()">＋ New Paper</button>
<button class="sideitem" onclick="scrollToId('upload')">📎 Upload PDF</button>
</aside>
<main class="main">
<section class="hero"><div class="wrap"><div class="badge"><span class="dot"></span>AI-Powered Academic Intelligence</div><h1>Understand Every<br>Question.</h1><p>Upload a question paper and let BloomAI extract questions, identify Bloom's Taxonomy, score difficulty, detect important and repeated questions, generate answers and build a focused study plan.</p></div></section>
<section class="wrap" id="upload"><div class="upload"><div class="uploadin"><div class="uploadicon">📄</div><h2>Upload your question paper</h2><p>Scanned and image-based PDFs are supported.</p><label class="filelabel" for="file">📎 Choose PDF</label><input id="file" type="file" accept="application/pdf,.pdf"><div class="filename" id="filename">No file selected</div><button class="primary" id="analyze">✨ Analyze with BloomAI</button></div></div></section>
<section class="progress wrap" id="progress"><div class="progressbox"><div class="progresshead"><b id="progressTitle">Preparing...</b><span id="progressPct">0%</span></div><div class="track"><div class="fill" id="progressFill"></div></div><div class="muted" id="progressSub" style="margin-top:9px"></div></div></section>
<div class="error wrap" id="error"></div>
<section class="ask wrap" id="ask"><div class="sectiontitle"><h2>🎙️ Ask BloomAI</h2><span class="muted">Type or speak · general AI by default</span></div><div class="askcard"><div class="askmode"><label for="askMode">Answer mode</label><select id="askMode"><option value="general">🌐 General AI</option><option value="paper">📄 Uploaded Paper</option></select><span class="muted">General mode works even without a paper.</span></div><div class="askrow"><textarea id="askText" class="asktext" placeholder="Try: Explain recursion in simple words..."></textarea><button class="mic" id="mic">🎙️</button><button class="askbtn" id="askBtn">Ask →</button></div><div class="status" id="status">Click the microphone and start speaking. Your speech can be used for general AI questions too.</div><div class="report-help" id="micHelp"><b>Microphone access is blocked.</b><br>In Chrome, click the site controls icon beside the address → <b>Site settings</b> → <b>Microphone</b> → <b>Allow</b>, then reload. If needed, open <b>http://localhost:8000</b> and allow the microphone there.</div><div class="answer" id="answer"></div><div class="answeractions" id="answerActions"><button class="tool" id="copy">Copy</button><button class="tool" id="speak">🔊 Read aloud</button><button class="tool" data-mode="simple">Simplify</button><button class="tool" data-mode="5mark">5-mark answer</button><button class="tool" data-mode="10mark">10-mark answer</button></div></div></section>
<section class="wrap dash" id="dashboard">
<div class="dashhead"><div><h2>Analysis Dashboard</h2><div class="muted" id="docinfo"></div></div><div class="dashactions"><button class="smallbtn" onclick="openQuiz()">🧪 Quiz Mode</button><button class="smallbtn" onclick="openStudy()">📅 Study Plan</button><button class="smallbtn" onclick="downloadReportPDF()">📄 Download PDF</button><button class="smallbtn" onclick="exportJSON()">↓ JSON</button></div></div>
<div class="stats"><div class="stat"><small>Questions</small><b id="sQ">—</b></div><div class="stat"><small>Papers</small><b id="sP">—</b></div><div class="stat"><small>Total Marks</small><b id="sM">—</b></div><div class="stat"><small>Avg Difficulty</small><b id="sD">—</b></div><div class="stat"><small>Paper Quality</small><b id="sQual">—</b></div></div>
<div class="grid2"><div class="panel" id="bloomPanel"><h3>🧠 Bloom's Taxonomy</h3><div id="bloomBars"></div></div><div class="panel" id="difficultyPanel"><h3>📈 Difficulty Distribution</h3><div id="diffBars"></div></div></div>
<div class="topicgrid"><div class="panel" id="topicPanel"><h3>📚 Topic Coverage</h3><div id="topicBars"></div></div><div class="panel"><h3>🔎 Papers Detected</h3><div class="papers" id="papers"></div></div></div>
<div class="grid2"><div class="panel"><h3>✨ AI Insights</h3><div class="list" id="insights"></div></div><div class="panel"><h3>⚠️ Potential Issues</h3><div class="list" id="issues"></div></div></div>
<div class="panel" id="importantPanel" style="margin-bottom:17px"><h3>⭐ Important Questions & Answers</h3><div class="list" id="importantQuestions"></div></div>
<div class="panel"><h3>🎯 Study Priorities</h3><div class="chips" id="priorities"></div></div>
<div class="panel" id="questionsPanel" style="margin-top:17px"><h3>📝 Question Bank</h3><div class="searchrow"><input id="search" class="search" placeholder="Search question, topic, year or Bloom level..."></div><div class="filters" id="filters"></div><div class="questions" id="questions"></div></div>
<div class="panel recommendations" id="recommendationsPanel" style="margin-top:17px"><div class="recommend-head"><div><h3 style="margin-bottom:5px">💡 Question Recommendations</h3><div class="muted">Add questions that match this paper's topics, level and Bloom coverage.</div></div></div><div class="recommend-controls"><label>Difficulty<select id="recDifficulty"><option value="">Match paper level</option><option>Easy</option><option>Moderate</option><option>Difficult</option><option>Very Difficult</option></select></label><label>Bloom level<select id="recBloom"><option value="">Auto-balance</option><option>Remember</option><option>Understand</option><option>Apply</option><option>Analyze</option><option>Evaluate</option><option>Create</option></select></label><label>Questions<select id="recCount"><option>4</option><option selected>6</option><option>8</option><option>10</option></select></label><button class="primary" id="recBtn" onclick="generateRecommendations()">✨ Recommend Questions</button></div><div class="recommend-grid" id="recommendations"><div class="empty">Analyze a paper to get paper-specific recommendations.</div></div></div>
</section>
<section class="wrap report" id="report"><h1>BloomAI Academic Analysis Report</h1><p id="reportIntro"></p><h2>Key Findings</h2><div id="reportFindings"></div><h2>Important Questions and Answers</h2><div id="reportImportant"></div><h2>Revision Priorities</h2><div id="reportPriorities"></div><h2>Question-wise Analysis</h2><div id="reportQuestions"></div></section>
<footer>BloomAI · AI-powered academic question paper analysis</footer>
</main></div>
<div class="modal" id="modal"><div class="modalbox"><div class="modalhead"><h2 id="modalTitle" style="margin:0"></h2><button class="smallbtn" onclick="closeModal()">Close</button></div><div class="modalbody" id="modalBody"></div></div></div>
<script>
const $=id=>document.getElementById(id);let analysis=null,lastAnswer='',lastFilename='',activeFilter='All';
function esc(v){return String(v??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#039;')}
function toggleTheme(){document.body.classList.toggle('light');localStorage.setItem('bloomai-theme',document.body.classList.contains('light')?'light':'dark')}
if(localStorage.getItem('bloomai-theme')==='light')document.body.classList.add('light');
function toggleSide(){$('side').classList.toggle('open')}
function scrollToId(id){const e=$(id);if(e)e.scrollIntoView({behavior:'smooth',block:'start'});$('side').classList.remove('open')}
function navTo(id){scrollToId(id)}
function showError(msg){$('error').textContent='⚠️ '+msg;$('error').style.display='block';scrollToId('error')}
function setStatus(msg){const el=$('status');if(el)el.textContent=msg}
function setProgress(p,title,sub){$('progress').style.display='block';$('progressFill').style.width=p+'%';$('progressPct').textContent=p+'%';$('progressTitle').textContent=title;$('progressSub').textContent=sub}
const file=$('file'),filename=$('filename'),analyze=$('analyze');file.addEventListener('change',()=>{if(!file.files.length){filename.textContent='No file selected';analyze.disabled=true;return}const f=file.files[0];if(!f.name.toLowerCase().endsWith('.pdf')){filename.textContent='Please select a PDF.';analyze.disabled=true;return}filename.textContent='✓ '+f.name;analyze.disabled=false});
let thinkingTimer=null;const generalThinking=['Understanding your question','Planning the answer','Writing the response','Almost ready'];const paperThinking=['Reading the paper','Finding relevant questions','Building the answer','Almost ready'];
function startThinking(){stopThinking();let i=0;const arr=(($('askMode')?.value)||'general')==='paper'?paperThinking:generalThinking;$('status').classList.add('thinking');$('status').textContent=arr[0];thinkingTimer=setInterval(()=>{$('status').textContent=arr[++i%arr.length]},650)}
function stopThinking(){if(thinkingTimer)clearInterval(thinkingTimer);thinkingTimer=null;$('status').classList.remove('thinking')}
analyze.addEventListener('click',async()=>{if(!file.files.length)return;$('error').style.display='none';analyze.disabled=true;setProgress(10,'Reading paper','Extracting text locally for faster analysis...');const fd=new FormData();fd.append('file',file.files[0]);try{await new Promise(r=>setTimeout(r,250));setProgress(28,'Reading every page','Text is extracted locally; scanned pages use fast vision OCR.');await new Promise(r=>setTimeout(r,250));setProgress(45,'Analyzing questions','Identifying sections, marks and question boundaries.');const r=await fetch('/analyze',{method:'POST',body:fd});const d=await r.json();if(!r.ok||!d.success)throw new Error(d.message||d.detail||'Analysis failed.');setProgress(72,'Understanding the paper','Classifying Bloom levels, difficulty and topics.');await new Promise(r=>setTimeout(r,300));analysis=d.analysis;lastFilename=d.filename||file.files[0]?.name||'';renderDashboard(analysis,d.filename);setProgress(100,'Complete','Your BloomAI dashboard is ready.');setTimeout(()=>{$('progress').style.display='none'},700)}catch(e){$('progress').style.display='none';showError(e.message)}finally{analyze.disabled=false}});
function bars(el,obj){const c=$(el);c.innerHTML='';const entries=Object.entries(obj||{});const total=entries.reduce((a,[,v])=>a+Number(v||0),0);entries.forEach(([k,v])=>{const pct=total?Number(v)/total*100:0;c.innerHTML+=`<div class="barrow"><div class="barinfo"><span>${esc(k)}</span><b>${v}</b></div><div class="bartrack"><div class="barfill" style="width:${pct}%"></div></div></div>`})}
function list(el,arr,empty='None detected'){const c=$(el);c.innerHTML=(arr?.length?arr:[empty]).map(x=>`<div class="item">${esc(x)}</div>`).join('')}
function renderDashboard(d,name){$('dashboard').style.display='block';$('docinfo').textContent=`${name} · ${d.paper_count||1} paper(s) · ${d.subject||'Question Paper'}`;$('sQ').textContent=d.total_questions??0;$('sP').textContent=d.paper_count??1;$('sM').textContent=d.total_marks??0;$('sD').textContent=d.average_difficulty||'Moderate';$('sQual').textContent=(d.paper_quality_score??0)+'/100';bars('bloomBars',d.bloom_distribution);bars('diffBars',d.difficulty_distribution);list('insights',d.insights);list('issues',d.issues);$('papers').innerHTML=(d.papers?.length?d.papers:[{label:'Question Paper',year:'',max_marks:d.total_marks||0,question_count:d.total_questions||0}]).map(p=>`<div class="item"><b>${esc(p.label||'Paper')}</b>${p.year?' · '+esc(p.year):''}<br>${p.question_count||0} questions · ${p.max_marks||0} marks</div>`).join('');$('priorities').innerHTML=(d.study_priorities?.length?d.study_priorities:['Review difficult and repeated questions.']).map(x=>`<span class="chip">🎯 ${esc(x)}</span>`).join('');$('topicBars').innerHTML=(d.topic_distribution||[]).slice(0,10).map(x=>{const max=Math.max(...(d.topic_distribution||[]).map(y=>Number(y.count||0)),1);return `<div class="barrow"><div class="barinfo"><span>${esc(x.topic)}</span><b>${x.count}</b></div><div class="bartrack"><div class="barfill" style="width:${Math.max(5,Number(x.count)/max*100)}%"></div></div></div>`}).join('')||'<div class="empty">No topic data.</div>';renderImportant();renderFilters();renderQuestions();$('recommendations').innerHTML='<div class="empty">Choose the difficulty/Bloom level and click Recommend Questions.</div>';buildReport();$('dashboard').scrollIntoView({behavior:'smooth',block:'start'})}
function renderImportant(){const arr=analysis?.important_questions||[];$('importantQuestions').innerHTML=(arr.length?arr:[...(analysis?.questions||[])].sort((a,b)=>(b.marks||0)-(a.marks||0)).slice(0,8)).map((x,i)=>`<div class="item"><b>${i+1}. ${esc(x.question)}</b><br><span class="muted">${esc(x.paper||'')} · ${x.marks||0} marks · ${esc(x.bloom_level||'')} · ${esc(x.difficulty||'')}</span><br><br>${esc(x.answer||'')}</div>`).join('')||'<div class="empty">No important questions available.</div>'}
function renderFilters(){const levels=['All','Remember','Understand','Apply','Analyze','Evaluate','Create'];$('filters').innerHTML=levels.map(x=>`<button class="filter ${activeFilter===x?'active':''}" onclick="setFilter('${x}')">${x}</button>`).join('')}
function setFilter(x){activeFilter=x;renderFilters();renderQuestions()}
$('search').oninput=renderQuestions;
function renderQuestions(){
  const c=$('questions');
  const q=analysis?.questions||[];
  const term=($('search').value||'').toLowerCase();
  const filtered=q.filter(x=>(activeFilter==='All'||x.bloom_level===activeFilter)&&(!term||JSON.stringify(x).toLowerCase().includes(term)));
  c.innerHTML='';
  if(!filtered.length){
    c.innerHTML='<div class="empty">No matching questions.</div>';
    return;
  }

  filtered.forEach((x,i)=>{
    const div=document.createElement('div');
    div.className='question';
    div.innerHTML=`
      <div class="qtop">
        <div class="qnum">${esc(x.question_number||i+1)}</div>
        <div class="qtext">${esc(x.question)}</div>
        <span class="tag">${esc(x.bloom_level||'')}</span>
        <span class="tag">${esc(x.difficulty||'')}</span>
      </div>
      <div class="qbody">
        <div class="qmeta">
          <div><b>Paper:</b> ${esc(x.paper||'')}</div>
          <div><b>Marks:</b> ${esc(x.marks||0)}</div>
          <div><b>Bloom:</b> ${esc(x.bloom_level||'')}</div>
          <div><b>Difficulty:</b> ${esc(x.difficulty||'')} (${esc(x.difficulty_score||0)}/100)</div>
          <div><b>Topics:</b> ${esc((x.topics||[]).join(', '))}</div>
        </div>
        <div class="existing-answer">
          <b>AI Answer</b>
          <div class="answertext">${esc(x.answer||'No answer generated.')}</div>
        </div>
        <div class="qtools">
          <button class="smallbtn" onclick="answerQuestionInline(this,'${encodeURIComponent(String(x.question||''))}','paper_simple','${esc(String(x.marks||''))}','${esc(String(x.bloom_level||''))}','${esc(String(x.difficulty||''))}','${encodeURIComponent((x.topics||[]).join(', '))}')">Explain simply</button>
          <button class="smallbtn" onclick="answerQuestionInline(this,'${encodeURIComponent(String(x.question||''))}','paper_5mark','5','${esc(String(x.bloom_level||''))}','${esc(String(x.difficulty||''))}','${encodeURIComponent((x.topics||[]).join(', '))}')">5-mark version</button>
          <button class="smallbtn" onclick="answerQuestionInline(this,'${encodeURIComponent(String(x.question||''))}','paper_10mark','10','${esc(String(x.bloom_level||''))}','${esc(String(x.difficulty||''))}','${encodeURIComponent((x.topics||[]).join(', '))}')">10-mark version</button>
        </div>
      </div>`;
    div.querySelector('.qtop').onclick=()=>div.classList.toggle('open');
    c.appendChild(div);
  });
}

async function answerQuestionInline(button, encodedQuestion, mode, marks, bloom, difficulty, encodedTopic){
  const card=button.closest('.question');
  const answerBox=card?.querySelector('.answertext');
  if(!answerBox)return;
  const question=decodeURIComponent(encodedQuestion||'');
  const topic=decodeURIComponent(encodedTopic||'');
  const original=button.textContent;
  document.querySelectorAll('.qtools .smallbtn').forEach(b=>b.disabled=true);

  const loadingText=mode==='paper_10mark'
    ? 'BloomAI is writing a detailed 10-mark answer…'
    : mode==='paper_5mark'
      ? 'BloomAI is writing a focused 5-mark answer…'
      : 'BloomAI is preparing a simple answer…';

  answerBox.dataset.previousAnswer=answerBox.textContent;
  answerBox.innerHTML='<span class="inline-thinking">'+loadingText+'</span>';
  button.textContent='Thinking…';

  try{
    const r=await fetch('/ask-stream',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        question,
        mode,
        marks: marks ? Number(marks) : null,
        bloom_level: bloom||'',
        difficulty: difficulty||'',
        topic
      })
    });
    if(!r.ok){
      const err=await r.json().catch(()=>({}));
      throw new Error(err.message||err.detail||'Unable to generate the answer.');
    }
    const reader=r.body?.getReader();
    if(!reader)throw new Error('Streaming response is not available.');
    const decoder=new TextDecoder();
    let answer='';
    while(true){
      const {value,done}=await reader.read();
      if(done)break;
      const chunk=decoder.decode(value,{stream:true});
      if(chunk.startsWith('__BLOOM_ERROR__'))throw new Error(chunk.replace('__BLOOM_ERROR__',''));
      answer+=chunk;
      answerBox.textContent=answer;
    }
    if(!answer.trim())throw new Error('No answer was returned.');
    answerBox.textContent=answer.trim();
  }catch(e){
    answerBox.textContent=answerBox.dataset.previousAnswer||'No answer could be generated.';
    const note=document.createElement('div');
    note.className='inline-error';
    note.style.marginTop='7px';
    note.textContent='⚠️ '+e.message;
    answerBox.appendChild(note);
  }finally{
    button.textContent=original;
    document.querySelectorAll('.qtools .smallbtn').forEach(b=>b.disabled=false);
  }
}


let recorder=null,recordingChunks=[],listening=false,mediaStream=null;
function showMicHelp(show=true){$('micHelp').classList.toggle('show',show)}
async function startRecording(){
  showMicHelp(false);
  try{
    if(!navigator.mediaDevices||!navigator.mediaDevices.getUserMedia)throw new Error('Microphone is not available in this browser.');
    mediaStream=await navigator.mediaDevices.getUserMedia({audio:true});
    const mime=MediaRecorder.isTypeSupported('audio/webm;codecs=opus')?'audio/webm;codecs=opus':'audio/webm';
    recorder=new MediaRecorder(mediaStream,{mimeType:mime});recordingChunks=[];
    recorder.ondataavailable=e=>{if(e.data.size)recordingChunks.push(e.data)};
    recorder.onstop=async()=>{
      listening=false;$('mic').classList.remove('listen');$('mic').textContent='🎙️';
      mediaStream?.getTracks().forEach(t=>t.stop());mediaStream=null;
      const blob=new Blob(recordingChunks,{type:recorder.mimeType||'audio/webm'});
      $('status').textContent='⚡ Transcribing...';
      try{
        const fd=new FormData();fd.append('audio',blob,'bloomai-voice.webm');
        const r=await fetch('/transcribe',{method:'POST',body:fd});const d=await r.json();
        if(!r.ok||!d.success)throw new Error(d.message||d.detail||'Transcription failed.');
        $('askText').value=d.text||'';
        if(!$('askText').value.trim())throw new Error('No speech was detected.');
        $('status').textContent='⚡ Speech captured — answering...';
        await ask('normal');
      }catch(e){$('status').textContent='⚠️ '+e.message}
    };
    recorder.start();listening=true;$('mic').classList.add('listen');$('mic').textContent='⏹';$('status').textContent='🎙️ Listening...';
  }catch(e){$('status').textContent='⚠️ '+e.message;showMicHelp(true)}
}
function stopRecording(){if(recorder&&recorder.state!=='inactive')recorder.stop()}
$('mic').onclick=()=>{if(listening)stopRecording();else startRecording()};
async function askPreset(q,mode){
  $('askText').value=q;
  const select=$('askMode');
  if(select) select.value='paper';
  scrollToId('ask');
  await ask(mode);
}
async function askRecommended(question,difficulty,bloom,marks){
  const q=String(question||'').trim();
  if(!q){showError('No recommended question was provided.');return}
  $('askText').value=q;
  const select=$('askMode');
  if(select) select.value='paper';
  scrollToId('ask');
  const numericMarks=Number(marks||5);
  const answerMode=numericMarks>=10?'paper_10mark':(numericMarks>=5?'paper_5mark':'paper_simple');
  await ask(answerMode);
}
$('askBtn').onclick=()=>ask('normal');document.querySelectorAll('.tool').forEach(b=>b.onclick=()=>ask(b.dataset.mode));$('askText').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();ask('normal')}});
async function ask(mode='normal'){
  const q=$('askText').value.trim();
  if(!q){$('status').textContent='Please type or speak a question.';return}
  const selectedMode=$('askMode').value||'general';
  const paperExplicit=/^paper_(simple|5mark|10mark|short|important)$/.test(mode);
const requestMode=paperExplicit
  ? mode
  : (['simple','5mark','10mark','short','important'].includes(mode)
      ? (selectedMode==='paper'?'paper_'+mode:mode)
      : selectedMode);
  $('error').style.display='none';
  $('askBtn').disabled=true;$('askBtn').textContent='Thinking';$('answer').style.display='block';$('answer').textContent='';$('answerActions').style.display='none';startThinking();
  try{
    const r=await fetch('/ask-stream',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q,mode:requestMode})});
    if(!r.ok){let d={};try{d=await r.json()}catch(_){ }throw new Error(d.message||d.detail||'Unable to answer.')}
    if(!r.body)throw new Error('Streaming is not supported by this browser.');
    const reader=r.body.getReader();const decoder=new TextDecoder();let answer='';let firstChunk=true;
    while(true){
      const {value,done}=await reader.read();
      if(done)break;
      const chunk=decoder.decode(value,{stream:true});
      if(chunk.startsWith('__BLOOM_ERROR__'))throw new Error(chunk.replace('__BLOOM_ERROR__','').trim());
      answer+=chunk;
      if(firstChunk){firstChunk=false;stopThinking();$('status').textContent='⚡ Answering...';$('answerActions').style.display='flex'}
      $('answer').textContent=answer;
    }
    if(!answer.trim())throw new Error('BloomAI returned an empty answer.');
    lastAnswer=answer.trim();$('answer').textContent=lastAnswer;$('answerActions').style.display='flex';$('status').textContent='✓ Answer ready.';
  }catch(e){
    stopThinking();$('answer').textContent='';$('answer').style.display='none';$('status').textContent='⚠️ '+e.message;
  }finally{$('askBtn').disabled=false;$('askBtn').textContent='Ask →'}
}
$('copy').onclick=()=>navigator.clipboard?.writeText(lastAnswer);$('speak').onclick=()=>{if('speechSynthesis'in window){speechSynthesis.cancel();speechSynthesis.speak(new SpeechSynthesisUtterance(lastAnswer))}};
function openModal(title,body){$('modalTitle').textContent=title;$('modalBody').innerHTML=body;$('modal').classList.add('show')}function closeModal(){$('modal').classList.remove('show')}
async function openQuiz(){if(!analysis)return showError('Analyze a paper first.');openModal('🧪 Quiz Mode','<div class="status thinking">BloomAI is creating your quiz</div>');try{const r=await fetch('/quiz',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({count:10})});const d=await r.json();if(!r.ok||!d.success)throw new Error(d.message||d.detail||'Quiz generation failed.');let html='<div id="quizbox">';d.quiz.forEach((x,i)=>{html+=`<div class="quizq"><b>${i+1}. ${esc(x.question)}</b>${x.options.map((o,j)=>`<label class="option"><input type="radio" name="q${i}" value="${j}"> ${esc(o)}</label>`).join('')}<div class="muted" id="exp${i}" style="display:none;margin-top:10px"></div></div>`});html+='<button class="primary" onclick="gradeQuiz()">Check Score</button><div id="score" class="score"></div></div>';$('modalBody').innerHTML=html;window.quizData=d.quiz}catch(e){$('modalBody').innerHTML='<div class="item">⚠️ '+esc(e.message)+'</div>'}}
async function gradeQuiz(){
  let score=0;
  window.quizData.forEach((x,i)=>{
    const s=document.querySelector(`input[name=q${i}]:checked`);
    const ok=s&&Number(s.value)===Number(x.answer_index);
    if(ok)score++;
    const ex=$('exp'+i);
    ex.style.display='block';
    ex.textContent=(ok?'✓ ':'✗ ')+'Correct answer: '+(x.options[x.answer_index]||x.explanation||'');
  });
  const total=window.quizData.length;
  const pct=Math.round(score/total*100);
  $('score').innerHTML=`<b>Quiz submitted: ${score}/${total} · ${pct}%</b>
    <div style="margin-top:12px">
      <button class="primary" id="downloadQuizBtn" type="button">⬇ Download Quiz PDF</button>
    </div>`;
  const btn=$('downloadQuizBtn');
  btn.onclick=async()=>{
    btn.disabled=true;btn.textContent='Preparing PDF…';
    try{
      const r=await fetch('/quiz-pdf',{
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({
          quiz:window.quizData,
          score,
          total,
          source:lastFilename||'Uploaded Question Paper'
        })
      });
      if(!r.ok){
        let msg='Could not create the quiz PDF.';
        try{const d=await r.json();msg=d.message||msg}catch(_){}
        throw new Error(msg);
      }
      const blob=await r.blob();
      const url=URL.createObjectURL(blob);
      const a=document.createElement('a');
      a.href=url;
      a.download='BloomAI_Quiz.pdf';
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(()=>URL.revokeObjectURL(url),1000);
      btn.textContent='✓ Quiz PDF Downloaded';
    }catch(e){
      btn.disabled=false;
      btn.textContent='⬇ Download Quiz PDF';
      $('score').insertAdjacentHTML('beforeend',`<div class="muted" style="margin-top:8px">⚠️ ${esc(e.message)}</div>`);
    }
  };
}


async function generateRecommendations(){
  if(!analysis){showError('Analyze a paper first.');return}
  const btn=$('recBtn');const grid=$('recommendations');
  btn.disabled=true;btn.textContent='Generating…';
  grid.innerHTML='<div class="item thinking">Matching the paper level, topics and Bloom coverage</div>';
  try{
    const r=await fetch('/recommendations',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({count:Number($('recCount').value||6),difficulty:$('recDifficulty').value,bloom_level:$('recBloom').value})});
    const d=await r.json();if(!r.ok||!d.success)throw new Error(d.message||d.detail||'Could not generate recommendations.');
    grid.innerHTML=(d.recommendations||[]).map((x,i)=>`<div class="recommend-card" style="animation-delay:${i*55}ms"><b>${i+1}. ${esc(x.question)}</b><div class="recommend-meta"><span>${esc(x.difficulty)}</span><span>${esc(x.bloom_level)}</span><span>${esc(x.suggested_marks)} marks</span><span>${esc(x.topic)}</span></div><div class="muted recommend-reason">${esc(x.reason)}</div></div>`).join('')||'<div class="empty">No recommendations available.</div>';
  }catch(e){grid.innerHTML='<div class="item">⚠️ '+esc(e.message)+'</div>'}
  finally{btn.disabled=false;btn.textContent='✨ Recommend Questions'}
}
async function openStudy(){if(!analysis)return showError('Analyze a paper first.');openModal('📅 Study Plan','<div class="status thinking">BloomAI is preparing your revision plan</div>');try{const r=await fetch('/study-plan',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({focus:'all'})});const d=await r.json();if(!r.ok||!d.success)throw new Error(d.message||d.detail||'Study plan failed.');const p=d.plan;let html=`<h3>${esc(p.summary||'Your personalized plan')}</h3><h4>Priorities</h4>${(p.priorities||[]).map(x=>`<div class="item">🎯 ${esc(x)}</div>`).join('')}<h4>Day 1</h4>${(p.day_1||[]).map(x=>`<div class="item">📅 ${esc(x)}</div>`).join('')}<h4>Day 2</h4>${(p.day_2||[]).map(x=>`<div class="item">📅 ${esc(x)}</div>`).join('')}<h4>Day 3</h4>${(p.day_3||[]).map(x=>`<div class="item">📅 ${esc(x)}</div>`).join('')}<h4>Quick Revision</h4>${(p.quick_revision||[]).map(x=>`<div class="item">⚡ ${esc(x)}</div>`).join('')}`;$('modalBody').innerHTML=html}catch(e){$('modalBody').innerHTML='<div class="item">⚠️ '+esc(e.message)+'</div>'}}
function buildReport(){
  if(!analysis)return;
  const d=analysis;
  const important=(d.important_questions&&d.important_questions.length?d.important_questions:[...(d.questions||[])].sort((a,b)=>(b.marks||0)-(a.marks||0)).slice(0,8));
  const insights=(d.insights||[]).join(' ');
  const issues=(d.issues||[]).length?'Potential issues identified: '+d.issues.join(' '):'No major issues were identified from the supplied document.';
  const priorities=(d.study_priorities||[]).join(' ');
  const bloom=Object.entries(d.bloom_distribution||{}).map(([k,v])=>`${esc(k)}: ${v} question(s)`).join(', ');
  const diff=Object.entries(d.difficulty_distribution||{}).map(([k,v])=>`${esc(k)}: ${v} question(s)`).join(', ');
  const topics=(d.topic_distribution||[]).slice(0,15).map(x=>`${esc(x.topic)} (${x.count})`).join(', ');
  let html=`<div class="reportpreview">
    <h1>BloomAI Academic Analysis Report</h1>
    <p><b>${esc(lastFilename||'Uploaded Question Paper')}</b> · ${esc(d.subject||'Academic Question Paper')}</p>
    <h2>Overview</h2>
    <p>BloomAI analyzed <b>${d.total_questions||0}</b> questions across <b>${d.paper_count||1}</b> paper(s). The combined maximum marks identified were <b>${d.total_marks||0}</b>. The overall average difficulty was <b>${esc(d.average_difficulty||'Moderate')}</b> and the paper-quality indicator was <b>${d.paper_quality_score||0}/100</b>.</p>
    <h2>Key Findings</h2><p>${esc(insights||'No additional AI insights were returned.')}</p>
    <p>${esc(issues)}</p>
    <h2>Bloom's Taxonomy</h2><p>${bloom||'No Bloom data available.'}</p>
    <h2>Difficulty Distribution</h2><p>${diff||'No difficulty data available.'}</p>
    <h2>Topic Coverage</h2><p>${topics||'No topic data available.'}</p>
    <h2>Important Questions and Answers</h2>`;
  important.forEach((x,i)=>{html+=`<div class="rpq"><p><b>${i+1}. ${esc(x.question)}</b> (${x.marks||0} marks)</p><p class="rpmeta">Paper: ${esc(x.paper||'')} · Bloom: ${esc(x.bloom_level||'')} · Difficulty: ${esc(x.difficulty||'')}</p><p class="rpanswer">${esc(x.answer||'No answer generated.')}</p></div>`});
  html+=`<h2>Revision Priorities</h2><p>${esc(priorities||'Review difficult, repeated and high-mark questions first.')}</p><h2>Question-wise Analysis</h2>`;
  (d.questions||[]).forEach((x,i)=>{html+=`<div class="rpq"><p><b>${i+1}. ${esc(x.question)}</b></p><p class="rpmeta">Marks: ${x.marks||0} · Paper: ${esc(x.paper||'')} · Bloom: ${esc(x.bloom_level||'')} · Difficulty: ${esc(x.difficulty||'')} (${x.difficulty_score||0}/100)</p>${(x.topics||[]).length?`<p class="rpmeta">Topics: ${esc((x.topics||[]).join(', '))}</p>`:''}<p class="rpanswer"><b>Answer:</b> ${esc(x.answer||'No answer generated.')}</p></div>`});
  html+='</div>';
  return html;
}
function openReport(){
  downloadReportPDF();
}
function downloadReportPDF(){
  if(!analysis){showError('Analyze a paper first.');return}
  $('error').style.display='none';
  setStatus('Preparing your complete PDF report…');
  const a=document.createElement('a');
  a.href='/report-pdf?download=1&ts='+Date.now();
  a.download='BloomAI-Academic-Analysis-Report.pdf';
  a.style.display='none';
  document.body.appendChild(a);
  a.click();
  a.remove();
  setStatus('✓ PDF download started. Check your browser Downloads folder.');
}

function downloadReportHTML(){
  const html=`<!doctype html><html><head><meta charset="utf-8"><title>BloomAI Academic Analysis Report</title><style>body{font-family:Arial,sans-serif;max-width:900px;margin:40px auto;padding:0 25px;color:#222;line-height:1.75}h1{font-size:30px}h2{color:#5b21b6;margin-top:28px}.rpq{padding:14px 0;border-bottom:1px solid #ddd}.rpmeta{font-size:12px;color:#666}.rpanswer{white-space:pre-wrap}@media print{body{margin:20px}}</style></head><body>${buildReport().replace(/^<div class="reportpreview">|<\/div>$/g,'')}</body></html>`;
  const blob=new Blob([html],{type:'text/html'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='BloomAI-Academic-Analysis-Report.html';document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(a.href),1500);
}
function printReportPreview(){
  const preview=buildReport();
  if(!preview){alert('Please analyze a paper first.');return}
  const w=window.open('', '_blank', 'width=1000,height=800');
  if(!w){
    if(confirm('The browser blocked the print preview window. Download the report as HTML instead?')) downloadReportHTML();
    return;
  }
  const clean=preview.replace(/^<div class="reportpreview">|<\/div>$/g,'');
  w.document.open();
  w.document.write(`<!doctype html><html><head><meta charset="utf-8"><title>BloomAI Academic Analysis Report</title><style>body{font-family:Arial,sans-serif;max-width:900px;margin:30px auto;padding:0 25px;color:#222;line-height:1.75}h1{font-size:30px}h2{color:#5b21b6;margin-top:28px}.rpq{padding:14px 0;border-bottom:1px solid #ddd}.rpmeta{font-size:12px;color:#666}.rpanswer{white-space:pre-wrap}@media print{body{margin:15px}h2{break-after:avoid}.rpq{break-inside:avoid}}
/* Final responsive spacing + grounded-action polish */
.recommend-controls{
  display:grid;
  grid-template-columns:minmax(170px,1fr) minmax(170px,1fr) minmax(120px,.7fr) minmax(220px,auto);
  gap:16px;
  align-items:end;
  margin-top:18px;
  margin-bottom:22px;
  padding:4px 0;
}
.recommend-controls label{
  display:flex;
  flex-direction:column;
  gap:7px;
  min-width:0;
  color:var(--muted);
  font-size:12px;
  font-weight:700;
}
.recommend-controls select{
  width:100%;
  min-height:42px;
  padding:9px 34px 9px 12px;
  border:1px solid var(--line);
  border-radius:12px;
  background:var(--card2);
  color:var(--text);
  font:inherit;
  outline:none;
  color-scheme:dark;
}
body.light .recommend-controls select{
  background:#fff;
  color:var(--text);
  color-scheme:light;
}
.recommend-controls select:focus{
  border-color:var(--p);
  box-shadow:0 0 0 3px rgba(46,168,255,.12);
}
.recommend-controls .primary{
  min-height:42px;
  white-space:nowrap;
}
.recommend-grid{display:grid;gap:14px}
.recommend-card{
  padding:18px 20px;
  border:1px solid var(--line);
  border-radius:16px;
  min-width:0;
}
.recommend-card>b{
  display:block;
  line-height:1.55;
  margin-bottom:11px;
  overflow-wrap:anywhere;
}
.recommend-meta{
  display:flex;
  flex-wrap:wrap;
  gap:7px;
  margin-bottom:10px;
}
.recommend-meta span{
  display:inline-flex;
  align-items:center;
  min-height:25px;
  padding:4px 9px;
  border-radius:999px;
  border:1px solid var(--line);
  background:rgba(46,168,255,.055);
  color:var(--muted);
  font-size:11px;
  white-space:nowrap;
}
.qtools{
  display:flex;
  flex-wrap:wrap;
  gap:9px;
  margin-top:16px;
}
.qtools .smallbtn{
  min-height:38px;
  white-space:nowrap;
}
.question .qbody{overflow-wrap:anywhere}
.question .qbody b{color:var(--text)}
.askcard{overflow:hidden}
.askrow{
  display:grid;
  grid-template-columns:minmax(0,1fr) 48px 112px;
  gap:10px;
  align-items:stretch;
}
.asktext{min-width:0;width:100%}
.qmeta{
  display:grid;
  grid-template-columns:repeat(2,minmax(0,1fr));
  gap:7px 28px;
  margin:2px 0 18px;
  line-height:1.65;
  color:var(--muted);
}
.existing-answer{
  padding:10px 0 0;
  line-height:1.62;
  overflow-wrap:anywhere;
}
.existing-answer>b{
  display:block;
  margin-bottom:7px;
  color:var(--text);
}
.answertext{
  color:var(--muted);
  white-space:pre-wrap;
}
.qinline-answer{display:none}
.inline-answer-text{color:var(--text);white-space:pre-wrap}
.inline-thinking{color:var(--p)}
.inline-error{color:var(--r)}
.qtools .smallbtn:disabled{
  opacity:.55;
  cursor:wait;
}
@media(max-width:900px){
  .recommend-controls{grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}
  .recommend-controls .primary{grid-column:1/-1;width:100%}
  .qmeta{grid-template-columns:1fr;gap:5px}
}
@media(max-width:600px){
  .recommend-controls{grid-template-columns:1fr;gap:12px;margin-bottom:18px}
  .recommend-controls .primary{grid-column:auto;width:100%}
  .recommend-card{padding:15px;border-radius:14px}
  .recommend-meta span{font-size:10px}
  .qmeta{font-size:13px}
  .qtools{display:grid;grid-template-columns:1fr}
  .qtools .smallbtn{width:100%}
  .askrow{grid-template-columns:1fr 48px}
  .askrow .askbtn{grid-column:1/-1;width:100%}
}

</style></head><body>${clean}</body></html>`);
  w.document.close();
  w.focus();
  setTimeout(()=>w.print(),700);
}
function exportJSON(){if(!analysis)return showError('Analyze a paper first.');const blob=new Blob([JSON.stringify(analysis,null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='BloomAI-analysis.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}
function newAnalysis(){analysis=null;file.value='';filename.textContent='No file selected';analyze.disabled=true;$('dashboard').style.display='none';$('report').style.display='none';scrollToId('upload')}
</script>
</body></html>'''


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
