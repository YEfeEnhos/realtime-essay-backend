from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from openai import OpenAI
import pdfplumber
import os
import random
import logging
logging.basicConfig(level=logging.INFO)

# --- Initialize OpenAI ---
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- FastAPI setup ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Preset Themes (leave empty for now) ---
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

# --- Preset Questions (leave empty for now) ---
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

# --- Configuration ---
MAX_CHAR_HISTORY = 4000
MAX_TURNS = 8

# --- Data Schema ---
class QuestionRequest(BaseModel):
    track: str
    cv_text: str
    history: list
    is_rapid_fire: bool
    theme_counts: dict = {}
    current_theme: str = ""
    academic_fields: list = []

# --- Utility: History Trimming ---
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

# --- Endpoint: Upload CV ---
@app.post("/upload-cv")
async def upload_cv(file: UploadFile = File(...)):
    contents = await file.read()
    with open("temp_cv.pdf", "wb") as f:
        f.write(contents)
    with pdfplumber.open("temp_cv.pdf") as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return {"text": text}

# --- Endpoint: Get Next Question ---
@app.post("/next-question")
async def next_question(req: QuestionRequest):
    conversation_history = smart_conversation_history(req.history)
    conversation_history = conversation_history or "This is the first question."
    track_questions = PRESETS.get(req.track, [])
    selected_preset = random.choice(track_questions) if track_questions else ""
    
    

    # Pick the prompt based on the phase
    if req.is_rapid_fire:
        prompt = f"""
You are a warm, perceptive assistant to a college counselor. Your job is to help the counselor gather **factual information** about the student’s academic background during this **rapid-fire round** of an interview.

Your job is only to gather **objective details** and follow the given instructions.




CV of the student:
{req.cv_text}

Conversation so far:
{conversation_history}

To understand which step you are in check the conversation history and the last question you asked. 

Steps to follow (instructions):
1. Begin with:
   Looks like [Extract 2-3 broad academic interests from the students CV] are your main academic interests. Could you tell me about three or four of your favourite subjects, related or unrelated to those interests?


2. When the student tells their academic interests choose one of them and ask: “How have you pursued [subject name] subject at school or during summer school?” If possible, use the CV and instead ask: “Looks like you studied this at [From the CV mention maximum 3 courses the student took]. Tell me more?”

   Then ask: “Have you done any research, internships or outisde of class activities related to [subject name] subject outside class?” If possible, use the CV and instead ask:“I especially would like to know more about [From the CV select 2-3 research, internships or outisde of class activities related to the current academic field of discussion]? Tell me more about them.”
   
   Then ask: “Is there anything more you want to add regarding this subject? If not lets move on.”  
    
   If the student says yes, ask: "What else would you like to add?” If the student says no, repeat the process by choosing the next subject which you can find in the conversation history. If you have covered all subjects, ask: “Thank you. I now have enough information to move on to broader questions if you have nothing to add.”


⚠️ Important:
- Ask ONE simple and factual question at a time.
- Do not put Q: at the begining of the question.
- Only ask about the singular subject at hand. (i.e. finish the discussion about one subject before moving to the next.)
- Stay focused on information-gathering only.
"""
  # <- use your rapid-fire prompt here
    else:
        prompt = f"""
You are a warm, perceptive assistant to a college counselor. The college counselor has asked you to interview the student, taking the preset questions as a starting point. The college counselor will use the interview transcript to brainstorm potential college application essay topics with the student.

Your task is to:
a. Gather as much detail as possible about the student’s academic interests, extracurricular involvement and personal background. These details are necessary for the counselor.
b. Build on these details with further questions about the student’s motivation and character as it relates to the subject being discussed.
c. Search for potential themes in the student’s answers, taking the preset themes as a starting point, and expanding upon these if another theme not covered in the preset theme list emerges. Further themes can range from intimate, inner‑growth narratives to large‑scale activism and intellectual exploration.

Student's CV:
{req.cv_text}

Interview Track: {req.track}

Preset question to base your next move on:
"{selected_preset}"

Conversation so far:
{conversation_history}

Themes discussed and their counts:
{req.theme_counts}

Current theme under discussion: {req.current_theme or 'None'}

List of preset themes:
{chr(10).join(PRESET_THEMES)}

Important Rules:
- Ask at most TWO questions per theme. After two, switch to a new theme.
- NEVER repeat a topic already deeply discussed.
- Build naturally based on student's previous answers.
- If a question has already been covered, invent a *better* one exploring deeper insight, without losing relevance.
- Stay strictly related to the selected track unless a powerful personal connection emerges.
- Phrase your questions conversationally, like a real human counselor talking warmly to a student.
- Prefer open-ended questions that encourage reflection and storytelling.
- Only output ONE question, no lists or options.

Reminder:
- Stay human, curious, and perceptive.
- Adapt wording naturally using clues from the CV and past conversation.
"""
  # <- use your regular prompt here

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a college essay coach helping a student discover themes."},
            {"role": "user", "content": prompt}
        ]
    )
    question = response.choices[0].message.content.strip()

    guessed_theme = ""
    theme_counts = req.theme_counts or {}

    # Only guess theme in regular phase
    if not req.is_rapid_fire:
        theme_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a classifier that identifies essay themes from conversation."},
                {"role": "user", "content": f"Given this conversation:\n{conversation_history}\n\nPick one most relevant theme from this list:\n{chr(10).join(PRESET_THEMES)}"}
            ]
        )
        guessed_theme = theme_response.choices[0].message.content.strip()
        theme_counts[guessed_theme] = theme_counts.get(guessed_theme, 0) + 1

    return {
        "question": question,
        "current_theme": guessed_theme,
        "theme_counts": theme_counts,
    }

# --- Endpoint: Speak ---
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
        return StreamingResponse(
            speech.iter_bytes(),
            media_type="audio/mpeg"
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Speech generation failed: {str(e)}"}
        )

# --- Endpoint: Transcribe Audio ---
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    audio = await file.read()
    with open("temp_audio.wav", "wb") as f:
        f.write(audio)
    with open("temp_audio.wav", "rb") as f:
        transcript = client.audio.transcriptions.create(model="whisper-1", file=f)
    return {"text": transcript.text}
