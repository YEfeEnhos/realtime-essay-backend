from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from openai import OpenAI
import pdfplumber
import random
import os

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Placeholder Preset Interview Themes and Questions ---
PRESET_THEMES = [
    "Overcoming rigid expectations & redefining success",
    "Heritage & family history as a source of purpose",
    "Immigrant / refugee identity & cultural adaptation",
    "Venturing beyond the comfort‑zone (geographic or personal)",
    "Evolving concept of home & belonging",
    "Interdisciplinary curiosity — bridging disparate fields",
    "Craftsmanship / entrepreneurship as self‑expression",
    "Social‑justice & advocacy (racism, refugees, education equity)",
    "Leadership / mentoring younger peers",
    "Resilience in the face of personal adversity",
    "Mind–body wellbeing & self‑care",
    "Intrinsic love of learning & intellectual independence",
    "Seeing patterns & connections in everyday life",
    "Purpose, legacy & impact‑driven research",
    "Creativity as personal voice",
    "Embracing uncertainty & adaptability",
    "Privilege, gratitude & 'giving back'",
    "Identity & self‑worth beyond external validation"
]

PRESETS = {
    "Academic Interests": [
        "What are your main academic interests? Could you tell me about three or four of your favourite subjects?",
        "Why do you like these subjects? What got you interested in them?",
        "Go subject by subject and tell me more about how you have pursued this interest recently at school or during summer school.",
        "How have you pursued this subject outside of the classroom? Have you done any internships or research projects?",
        "Is there anything about the way this subject is taught that works for you or doesn’t work for you?",
        "Do any of these subjects relate to what you think you want to study at university?",
        "Do you know what you want to study at university?",
        "If you don’t know, why don’t you know?",
        "What do you think you want to do after you graduate? Why do you want this career?",
        "Besides those things you have mentioned already, in what other ways are you preparing yourself for this career?",
        "Why do you want to study in the United States?",
        "How much freedom do you have in choosing your university subject or your career?",
        "Do you have any intellectual interests or ideas that you are deeply absorbed in or fascinate you? Could you give me one or two examples?",
        "Tell me about each in order. Why do these fascinate you? What do you want to know more about?",
        "Have you had any obstacles or challenges in your academic life that affected your academic results?",
        "How have you dealt with these or are currently dealing with these?",
        "Has anything or anyone helped you overcome these challenges?",
        "Is there anything else about your academic interests that I haven’t asked you about that you think you would like to share?"
    ],
    "Extracurricular Activities": [
        "Let’s start by listing your most important extracurricular activities.",
        "Is there anything else you might be forgetting? Have a look at your CV or list of activities if you’d like.",
        "Now go through each activity one by one and briefly tell me: more about how you have pursued it, and if you have a specific role in it.",
        "What you enjoy about this activity and what it brings you.",
        "Why do you do this activity, and why do you care about it?",
        "How did you hear about it and what led you to sign up?",
        "What specifically do you do? What is your role?",
        "What is your particular strength in this area, what do you in particular bring?",
        "What have you found most challenging about this work?",
        "What have you learnt about yourself and others from doing it?",
        "What have you found most rewarding about it?",
        "Do you see yourself continuing it in the future?",
        "Do you have any anecdotes about an activity that you might want to share? Anything that stands out to you?"
    ],
    "Family & Background": [
        "How do your friends or people closest to you describe you?",
        "Do you agree?",
        "Which parts of your character do you like, and which parts do you wish you could change?",
        "Tell me about your family. This can be your immediate family or you can also talk about your extended family if they are important to you.",
        "Who is your favourite person in your family? Tell me more about your relationship with them.",
        "Is there anyone you clash with and why?",
        "What do you think are your family’s values and do you agree with them?",
        "How would you describe your socioeconomic, national, ethnic or faith background?",
        "What does your background mean to you? Do you think this defines you or not really? And if not, why not.",
        "Have you lived in the same place your whole life, or have you moved for any reason?",
        "If you did move, was the change easy or difficult?",
        "How have the places where you have lived affected your identity?",
        "Where is home for you? How do you define home?",
        "Could you tell me a memory about home or about growing up? Something that has stuck with you over the years.",
        "What does this memory tell you about your childhood?",
        "Did your parents (or grandparents) go to university? Where? What did they study?",
        "What is your gender or sexual identity, if you feel like sharing?",
        "Is your gender or sexual identity important to you? How has it informed your perspective?",
        "Are there any obstacles you have struggled with or overcome in your personal or family life/as a community?",
        "Are you engaged with politics and/or activism or do you stay out of it?",
        "If you are engaged, what issues do you feel most passionate about?",
        "What does activism mean to you, and how have you been involved?",
        "If you stay out of politics, why is this the case and is this important to you?",
        "How has your upbringing informed who you are today and how you see your future?",
        "Do you have any worries about your future?",
        "Try to imagine yourself sitting in a small university classroom and having a discussion about your favourite subject. What perspective do you think you will bring?",
        "What, beyond academics, do you hope to gain from attending university in the US?",
        "Is there anything else that is really important to you, to your understanding of yourself or other people’s understanding of you that you haven’t had a chance to talk about yet?"
    ]
}

MAX_CHAR_HISTORY = 3000
MAX_TURNS = 5


# --- Request Models ---
class QuestionRequest(BaseModel):
    track: str
    cv_text: str
    history: list  # list of {"question": ..., "answer": ...}
    theme_counts: dict = {}
    current_theme: str = ""
    academic_fields: list = []


# --- Utilities ---
def smart_conversation_history(history):
    recent_history = history[-MAX_TURNS:]
    text = "\n".join([f"Q: {turn['question']}\nA: {turn['answer']}" for turn in recent_history])
    if len(text) > MAX_CHAR_HISTORY:
        for i in range(len(recent_history)):
            trimmed = "\n".join([f"Q: {t['question']}\nA: {t['answer']}" for t in recent_history[i:]])
            if len(trimmed) <= MAX_CHAR_HISTORY:
                return trimmed
        return ""
    return text


# --- Endpoints ---
@app.post("/upload-cv")
async def upload_cv(file: UploadFile = File(...)):
    contents = await file.read()
    with open("temp_cv.pdf", "wb") as f:
        f.write(contents)

    with pdfplumber.open("temp_cv.pdf") as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

    # Extract academic fields
    try:
        field_resp = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You extract academic fields from CVs."},
                {"role": "user", "content": f"From this CV, list 2 or 3 academic fields (like computer science, math, biology) the student is most focused on:\n{text}"}
            ]
        )
        field_list = field_resp.choices[0].message.content.strip()
    except Exception as e:
        field_list = "Not available"

    return {"text": text, "fields": field_list}


@app.post("/next-question")
async def next_question(req: QuestionRequest):
    track_questions = PRESETS.get(req.track, [])
    selected_preset = random.choice(track_questions) if track_questions else ""
    conversation_history = smart_conversation_history(req.history)
    conversation_history = conversation_history or "This is the first question."

    # --- Theme Identification from Last Exchange ---
    last_turn = req.history[-1] if req.history else None
    last_exchange = f"Q: {last_turn['question']}\nA: {last_turn['answer']}" if last_turn else ""

    theme_response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a theme classifier."},
            {"role": "user", "content": f"Identify the most relevant college essay theme in this Q/A:\n{last_exchange}\n\nPick from this list:\n{chr(10).join(PRESET_THEMES)}\nReturn one theme or 'None'."}
        ]
    )
    guessed_theme = theme_response.choices[0].message.content.strip()
    if guessed_theme.lower() == "none":
        guessed_theme = ""

    theme_counts = req.theme_counts or {}
    if guessed_theme:
        theme_counts[guessed_theme] = theme_counts.get(guessed_theme, 0) + 1

    must_switch_theme = theme_counts.get(guessed_theme, 0) >= 2

    theme_instruction = f"""
Themes so far: {theme_counts}
Current theme: {guessed_theme or 'None'}

{"Switch to a new unexplored theme." if must_switch_theme else "Continue on current theme if possible."}
"""

    # --- Prompt Assembly ---
    prompt = f"""
You are a warm, perceptive assistant to a college counselor. The counselor has asked you to interview the student.

Your goals:
- Ask about the student’s academic interests, extracurriculars, personal background.
- Ask follow-ups that reveal motivation and character.
- Discover college essay themes that fit the student.

Student's CV:
{req.cv_text}

Academic fields of interest:
{", ".join(req.academic_fields) if req.academic_fields else "N/A"}

Track: {req.track}
Preset question for context:
"{selected_preset}"

Conversation so far:
{conversation_history}

{theme_instruction}

Guidelines:
- Ask only ONE question at a time.
- NEVER repeat what has already been discussed.
- Don't say "theme" or list themes.
- Prefer open-ended questions.
- Adapt to the student’s tone and academic interests.
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a perceptive college essay interviewer."},
            {"role": "user", "content": prompt}
        ]
    )
    question = response.choices[0].message.content.strip()

    return {
        "question": question,
        "current_theme": guessed_theme,
        "theme_counts": theme_counts
    }


@app.post("/speak")
async def speak_text(request: dict):
    text = request.get("text")
    if not text:
        return {"error": "No text provided."}

    try:
        speech = client.audio.speech.create(
            model="tts-1",
            input=text,
            voice="nova",
            response_format="mp3"
        )
        return StreamingResponse(speech.iter_bytes(), media_type="audio/mpeg")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"TTS failed: {str(e)}"})


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    audio = await file.read()
    with open("temp_audio.wav", "wb") as f:
        f.write(audio)

    with open("temp_audio.wav", "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )

    return {"text": transcript.text}
