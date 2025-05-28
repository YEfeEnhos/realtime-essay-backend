from fastapi import FastAPI, UploadFile, File # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from fastapi.responses import StreamingResponse, JSONResponse # type: ignore
from pydantic import BaseModel # type: ignore
from openai import OpenAI # type: ignore
import pdfplumber # type: ignore
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
        "How do your friends or people closest to you describe you? Do you agree?",
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
        "What does activism mean to you, how have you been involved, and hat issues do you feel most passionate about? If you stay out of politics, why is this the case and is this important to you?",
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
    academic_fields: list = []  # For favorite subjects
    extracurricular_fields: list = []  # For top 5 activities
    background_index: int = 0

# --- Utility: History Trimming ---
def smart_conversation_history(history):
    if not history:
        return "This is the first question."

    # Include all past questions (without answers)
    all_questions = "\n".join([f"Q: {turn['question']}" for turn in history[:-1]])
    last_q_and_a = f"Q: {history[-1]['question']}\nA: {history[-1]['answer']}"

    return f"{all_questions}\n{last_q_and_a}".strip()


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
    # --- Extract structured info based on tags ---
    if req.history:
        last_tag = req.history[-1].get("tag", "")

        if req.track == "Academic Interests" and last_tag == "ask_fav_subjects":
            req.academic_fields = [s.strip() for s in req.history[-1]['answer'].split(",")]

        if req.track == "Extracurricular Activities" and last_tag == "ask_top_activities":
            req.extracurricular_fields = [s.strip() for s in req.history[-1]['answer'].split(",")]
    logging.info(f"Academic fields: {req.academic_fields}")
    conversation_history = conversation_history or "This is the first question."
    track_questions = PRESETS.get(req.track, [])
    selected_preset = random.choice(track_questions) if track_questions else ""
    themes = PRESET_THEMES
    tag = ""
    
    logging.info(f"track questions: {track_questions}")
    logging.info(f"themes: {themes}")
    
    logging.info(f"theme counts: {req.theme_counts}")
    
    

    # Pick the prompt based on the phase
    if req.is_rapid_fire and req.track == "Academic Interests":
        
    
        already_asked = any("three or four of your favourite subjects" in turn["question"].lower() for turn in req.history)
        if not req.academic_fields:
            if not already_asked:
                return {
                    "question": "Could you tell me about three or four of your favourite subjects, related or unrelated to those interests?",
                    "current_theme": "",
                    "theme_counts": req.theme_counts,
                    "tag": "ask_fav_subjects"
                }
            else:
                last_turn = req.history[-1]
                if last_turn.get("tag") == "ask_fav_subjects":
                     req.academic_fields = [s.strip() for s in last_turn["answer"].split(",")]
        discussed_fields = set()
        for turn in req.history:
            for field in req.academic_fields:
                if field.lower() in turn["question"].lower():
                    discussed_fields.add(field)
        
        remaining_fields = [f for f in req.academic_fields if f not in discussed_fields]
        if not remaining_fields:
            return {
                "question": "Thank you. I now have enough information to move on to broader questions if you have nothing to add.",
                "current_theme": "",
                "theme_counts": req.theme_counts,
                "tag": ""
            }
            
        current_field = remaining_fields[0]
        
        field_q_count = sum(
            1 for turn in req.history if current_field.lower() in turn["question"].lower()
        )
        
        cv_text = req.cv_text.lower() if req.cv_text else ""
        question = ""
        
        if field_q_count == 0:
            if cv_text:
                course_lines = [line for line in cv_text.splitlines() if current_field.lower() in line]
                courses = "; ".join(course_lines[:3]) if course_lines else ""
                if courses:
                    question = f"Looks like you studied {current_field} at {courses}. Tell me more about them or other in-school or summer courses you took part in."
                else:
                    question = f"How have you pursued {current_field} at school or during summer school?"
            else:
                question = f"How have you pursued {current_field} at school or during summer school?"
                
        elif field_q_count == 1:
            # Ask about outside class activities
            if cv_text:
                activity_lines = [line for line in cv_text.splitlines() if current_field.lower() in line and any(word in line for word in ["research", "intern", "project"])]
                activities = "; ".join(activity_lines[:2]) if activity_lines else ""
                if activities:
                    question = f"I especially would like to know more about {activities}. Tell me more about them or other internships, research, or outside class activities related to {current_field}."
                else:
                    question = f"Have you done any research, internships, or outside class activities related to {current_field}?"
            else:
                question = f"Have you done any research, internships, or outside class activities related to {current_field}?"

        elif field_q_count == 2:
            question = f"Is there anything more you want to add regarding {current_field}? If not, let's move on."

        else:
            # If more than 3 questions, mark it done
            return {
                "question": f"Thanks for sharing about {current_field}.",
                "current_theme": "",
                "theme_counts": req.theme_counts,
                "tag": ""
            }

        return {
            "question": question,
            "current_theme": "",
            "theme_counts": req.theme_counts,
            "tag": ""
        }
                    
        
    elif req.is_rapid_fire and req.track == "Extracurricular Activities":
        prompt = f"""
Your job is to gather **objective details** and follow the given instructions.

CV of the student:
{req.cv_text or "None provided."}

Conversation so far:
{conversation_history}

To understand which step you are in check the conversation history and ask the question accordingly. Make sure to ask the questions in the order below and not repeat.

Steps to follow (instructions):

Only do this if the CV is not provided.
“What extracurricular activities or clubs are you involved in? This could be sport, volunteer work, community engagement, arts/culture, or simply what you like doing in your free time. Could you start by listing your most important extracurricular activities?” (Here we aim to get a extensive list so encourage the student to list as many as possible.)

2. When student list their activities or provided a CV:
Depending on the list the student provides say, “To me [5 **most impressive and diverse** items from the ones student listed] stand out. What do you think? Could you pick the most important 5 activities you would like to talk about today?” (Ensure to highlight that they should be the most important ones.)
If the CV is provided, you may suggest the top 5 **most impressive and diverse** items, avoiding overlapping roles (e.g., two research projects).

3. Then ask:
“In which order would you like to talk about these five activities?”

4. For each activity (one at a time), ask the following **in sequence**, only moving to the next after the previous is answered. Stick with the same activity till step 5:

    4.1) Could you tell me more about this activity and how long you’ve done it? What’s your role in it? What do you bring to it personally?
    4.2) What do you enjoy about it? What’s been most rewarding?
    4.3) What have you found challenging about this work?
    4.4) What have you learned about yourself or others?
    4.5) Do you see yourself continuing it? If you’ve stopped or had to cut back (or will do in the future), how do you feel?
    4.6) Do you have any anecdotes, moments or take-aways that stand out? 
    
5. After all questions are asked for one activity:
- Ask: “Is there anything more you want to add regarding this activity? If not, let's move on.”

- If the student says yes, ask: "What else would you like to add?”
- If they say no, move to the next activity from the list, and repeat step 4.

6. When all activities have been covered:
“Thanks for sharing those. I now can move on or I can hear about another activity. If you want to talk about a new activity please state it or else say lets move on.”

- If the student says yes and states an activity, do step 4 again for that activity.
- If the student says no, say: “Thank you. That’s the end of the extracurricular interview!”

⚠️ Important:
- Look at the conversation history to determine the step and which activity you’re on.
- Ask only ONE factual question per turn.
- NEVER jump to another activity or step prematurely.
- Do not repeat or rephrase the same question if it has already been asked.
- Avoid putting 'Q:' in front of your question.
"""
    
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a warm, perceptive assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        question = response.choices[0].message.content.strip()

        if "most important 5 activities" in question.lower():
            tag = "ask_top_activities"

        return {
            "question": question,
            "current_theme": "",
            "theme_counts": req.theme_counts,
            "tag": tag
        }
    
    elif req.track == "Family & Background":
        all_bg_questions = PRESETS["Family & Background"]

        if req.background_index >= len(all_bg_questions):
            return {
                "question": "Thank you. That’s the end of the background interview!",
                "current_theme": "",
                "theme_counts": req.theme_counts,
            }

        next_question = all_bg_questions[req.background_index]
        last_answer = req.history[-1]["answer"] if req.history else ""

        gpt_prompt = f"""
    You are a warm, perceptive college counselor conducting an interview with a student. 

    Your goal is to ask the preset questions **one by one in the given order** from the “Family & Background” list. Do not invent new questions or reorder them. Add a short, friendly sentence that naturally reacts to the student’s **last answer**, and then ask the **next** question.

    Here is the student's previous answer:
    "{last_answer}"

    The next question to ask is:
    "{next_question}"

    Begin with a natural transition or reflection, and then ask the question in a conversational tone.
    """

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a friendly college counselor helping a student reflect on their background."},
                {"role": "user", "content": gpt_prompt}
            ]
        )

        q_text = response.choices[0].message.content.strip()
        tag = ""

        if "three or four of your favourite subjects" in q_text.lower():
            tag = "ask_fav_subjects"
        elif "most important 5 activities" in q_text.lower():
            tag = "ask_top_activities"

        return {
            "question": q_text,
            "current_theme": "",
            "theme_counts": req.theme_counts,
            "tag": tag
        }


    
    else:
        prompt = f"""
Your task is to:
a. Gather as much detail as possible about the student’s academic interests or extracurricular involvement or personal background (depending on the choosen track). These details are necessary for the counselor.
b. Build on these details with further questions about the student’s motivation and character as it relates to the subject being discussed.

Student's CV:
{req.cv_text}

If the student has not provided a CV pay more attention to conversation history and preset questions.

Interview Track: {req.track}

Preset question to base your next move on:
"{track_questions}"

If the academic track is choosen ask at least once about challanges and obstacles in the academic life of the student.

Pick the most relevant preset question from the list according to the conversation history and the CV.

Conversation so far:
{conversation_history}

Themes discussed and their counts:
{req.theme_counts}

Pick a relevant theme from the list of preset themes. This should help you give direction to the question.

List of preset themes:
{themes}

Instructions:
- Check the conversation history and theme counts to see what themes have been discussed.
- Ask at most TWO questions per theme. After two, switch to a new theme. To understand how many times the theme has been discussed check the theme counts and conversation history.
- Prioritize themes that have NOT yet been discussed.
- If this is not the first question, before generating a question check the conversation history to give 1-2 lines of reflection (something like that sounds interesting) to the students response and then ask the question.
- NEVER repeat a topic already deeply discussed.
- Build naturally based on student's previous answers.
- Stay strictly related to the selected track unless a powerful personal connection emerges.
- Phrase your questions conversationally, like a real human counselor talking warmly to a student.
- Prefer open-ended questions that encourage reflection and storytelling.
- Only output ONE question, no lists or options.
- Do not begin with "Q:".

Reminder:
- Stay human, curious, and perceptive.
- Adapt wording naturally using clues from the CV and past conversation.
"""
  # <- use your regular prompt here

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a warm, perceptive assistant to a college counselor. The college counselor has asked you to interview the student, taking the preset questions as a starting point. The college counselor will use the interview transcript to brainstorm potential college application essay topics with the student."},
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
        raw_theme = theme_response.choices[0].message.content.strip()
        guessed_theme = next(
            (theme for theme in PRESET_THEMES if theme in raw_theme),
            None
        )
        if guessed_theme:
            theme_counts[guessed_theme] = theme_counts.get(guessed_theme, 0) + 1
        else:
            logging.warning(f"Could not match theme in response: {raw_theme}")

    tag = ""
    if "three or four of your favourite subjects" in question.lower():
        tag = "ask_fav_subjects"
    elif "most important 5 activities" in question.lower():
        tag = "ask_top_activities"

    return {
        "question": question,
        "current_theme": guessed_theme,
        "theme_counts": theme_counts,
        "tag": tag
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
