from fastapi import FastAPI, HTTPException, Depends, Header, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from bson import ObjectId
from datetime import datetime, timezone, timedelta
from models import Note, Task, Image, Audio
from typing import List, Optional
from pydantic import BaseModel
import base64
import re
import traceback
from transformers import pipeline
import spacy
from dateutil.parser import parse as parse_date
import speech_recognition as sr
from pydub import AudioSegment
import io

app = FastAPI()
print("App instance created:", app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://orbit-3vs4yi4ip-rhythms-projects-e260ea03.vercel.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MONGO_URI = "mongodb+srv://orbituser:OrbitPass123!@orbit-cluster.4bi15tv.mongodb.net/?retryWrites=true&w=majority&appName=orbit-cluster"
client = MongoClient(MONGO_URI, server_api=ServerApi("1"))
try:
    client.server_info()
    print("Successfully connected to MongoDB Atlas!")
except Exception as e:
    print(f"Failed to connect to MongoDB Atlas: {str(e)}")
db = client["orbit"]
collection = db["canvas_elements"]

# Initialize Hugging Face question-answering pipeline
qa_pipeline = pipeline(
    "question-answering", model="distilbert-base-uncased-distilled-squad"
)
print("Loaded Hugging Face question-answering pipeline")

# Initialize spaCy for intent recognition and entity extraction
nlp = spacy.load("en_core_web_sm")
print("Loaded spaCy English model")

class UpdateElement(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    x: Optional[float] = None
    y: Optional[float] = None
    due_date: Optional[str] = None
    due_time: Optional[str] = None
    priority: Optional[str] = None
    repeat: Optional[str] = None
    completed: Optional[bool] = None
    last_reset: Optional[str] = None
    is_edited: Optional[bool] = None
    image_data: Optional[str] = None
    audio_data: Optional[str] = None
    transcription: Optional[str] = None  # Keep transcription field

def element_to_dict(element):
    element["_id"] = str(element["_id"])
    return element

@app.post("/notes/", response_model=dict)
def create_note(note: Note, user_id: str):
    note_dict = note.model_dump()
    note_dict["user_id"] = user_id
    note_dict["created_at"] = datetime.now(timezone.utc).isoformat()
    result = collection.insert_one(note_dict)
    note_dict["_id"] = str(result.inserted_id)
    return note_dict

@app.post("/tasks/", response_model=dict)
def create_task(task: Task, user_id: str):
    task_dict = task.model_dump()
    task_dict["user_id"] = user_id
    task_dict["created_at"] = datetime.now(timezone.utc).isoformat()
    result = collection.insert_one(task_dict)
    task_dict["_id"] = str(result.inserted_id)
    return task_dict

@app.post("/images/", response_model=dict)
def create_image(image: Image, user_id: str):
    image_dict = image.model_dump()
    image_dict["user_id"] = user_id
    image_dict["created_at"] = datetime.now(timezone.utc).isoformat()
    result = collection.insert_one(image_dict)
    image_dict["_id"] = str(result.inserted_id)
    return image_dict

@app.post("/audios/", response_model=dict)
def create_audio(audio: Audio, user_id: str):
    audio_dict = audio.model_dump()
    audio_dict["user_id"] = user_id
    audio_dict["created_at"] = datetime.now(timezone.utc).isoformat()

    # If audio_data is provided, transcribe it
    if audio_dict.get("audio_data"):
        try:
            # Decode base64 audio data
            audio_bytes = base64.b64decode(audio_dict["audio_data"].split(",")[1])
            audio_file = io.BytesIO(audio_bytes)

            # Convert to WAV format (required by SpeechRecognition)
            audio_segment = AudioSegment.from_file(audio_file)
            wav_file = io.BytesIO()
            audio_segment.export(wav_file, format="wav")
            wav_file.seek(0)

            # Transcribe the audio
            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_file) as source:
                audio_data = recognizer.record(source)
                transcription = recognizer.recognize_google(audio_data)
                audio_dict["transcription"] = transcription
        except Exception as e:
            print(f"Error transcribing audio: {str(e)}")
            audio_dict["transcription"] = None

    result = collection.insert_one(audio_dict)
    audio_dict["_id"] = str(result.inserted_id)
    return audio_dict

@app.get("/elements/", response_model=List[dict])
def get_elements(user_id: str):
    print("get_elements endpoint called with user_id:", user_id)
    elements = list(collection.find({"user_id": user_id}))
    return [element_to_dict(element) for element in elements]

@app.get("/elements/{element_id}", response_model=dict)
def get_element(element_id: str, user_id: str):
    element = collection.find_one({"_id": ObjectId(element_id), "user_id": user_id})
    if not element:
        raise HTTPException(status_code=404, detail="Element not found or unauthorized")
    return element_to_dict(element)

@app.put("/elements/{element_id}", response_model=dict)
def update_element(element_id: str, updated_data: UpdateElement, user_id: str):
    print("Received updated_data:", updated_data)
    updated_data_dict = updated_data.model_dump(exclude_unset=True)
    updated_data_dict["updated_at"] = datetime.now(timezone.utc).isoformat()
    result = collection.update_one(
        {"_id": ObjectId(element_id), "user_id": user_id}, {"$set": updated_data_dict}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Element not found or unauthorized")
    return {"message": "Element updated successfully"}

@app.delete("/elements/{element_id}")
def delete_element(element_id: str, user_id: str):
    result = collection.delete_one({"_id": ObjectId(element_id), "user_id": user_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Element not found or unauthorized")
    return {"message": "Element deleted successfully"}

@app.get("/search", response_model=List[dict])
def search_elements(user_id: str, query: str):
    try:
        if not query:
            return []

        query_lower = query.lower()
        print(f"Searching for user_id: {user_id}, query: {query_lower}")
        elements = list(collection.find({"user_id": user_id}))
        print(f"Found {len(elements)} elements for user_id: {user_id}")

        results = []
        for element in elements:
            # Search in title
            title = element.get("title", "")
            title_match = bool(
                title and isinstance(title, str) and query_lower in title.lower()
            )

            # Search in content (for notes)
            content = element.get("content", "")
            content_match = element.get("type") == "note" and bool(
                content and isinstance(content, str) and query_lower in content.lower()
            )

            # Search in created_at date
            created_at = element.get("created_at")
            date_match = False
            if created_at and isinstance(created_at, str):
                try:
                    if created_at.endswith("Z"):
                        created_at = created_at.replace("Z", "+00:00")
                    created_at = datetime.fromisoformat(created_at)
                    date_match = (
                        created_at.strftime("%B").lower().find(query_lower) != -1
                        or str(created_at.day) == query_lower
                        or str(created_at.year) == query_lower
                    )
                except ValueError as e:
                    print(f"Error parsing created_at: {e}")
                    date_match = False

            # Search in transcription (for audio)
            transcription = element.get("transcription", "")
            transcription_match = element.get("type") == "audio" and bool(
                transcription
                and isinstance(transcription, str)
                and query_lower in transcription.lower()
            )

            if title_match or content_match or date_match or transcription_match:
                results.append(element_to_dict(element))

        print(f"Returning {len(results)} results")
        return results
    except Exception as e:
        print(f"Error in /search endpoint: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/smart_search", response_model=dict)
def smart_search(user_id: str, query: str):
    try:
        if not query:
            return JSONResponse(
                content={"answer": "Please provide a query.", "elements": []},
                headers={"Access-Control-Allow-Origin": "http://localhost:5173"},
            )

        print(f"Smart search for user_id: {user_id}, query: {query}")
        # Step 1: Fetch user elements
        user_elements = list(collection.find({"user_id": user_id}))
        if not user_elements:
            return JSONResponse(
                content={"answer": "No elements found for this user.", "elements": []},
                headers={"Access-Control-Allow-Origin": "http://localhost:5173"},
            )

        # Step 2: Use spaCy to parse the query and determine intent
        doc = nlp(query.lower())
        intent = "unknown"
        entities = {"date": None, "priority": None, "keyword": None, "specific_info": None}

        # Extract entities (e.g., dates, keywords)
        for ent in doc.ents:
            if ent.label_ == "DATE":
                entities["date"] = ent.text
            elif ent.label_ in ["ORG", "PERSON"]:
                entities["keyword"] = ent.text

        # Determine intent based on keywords
        query_lower = query.lower()
        if "task" in query_lower:
            if "due" in query_lower or "upcoming" in query_lower:
                intent = "tasks_due"
            elif "tag" in query_lower or "tagged" in query_lower:
                intent = "tasks_priority"
                # Extract the priority (e.g., "urgent" in "tasks tagged 'urgent'")
                priority_match = re.search(r"tag(?:ged)?\s+['\"]([^'\"]+)['\"]", query_lower)
                if priority_match:
                    priority = priority_match.group(1).lower()
                    entities["priority"] = "high" if "urgent" in priority else priority
                else:
                    # Fallback: look for a word after "tagged"
                    words = query_lower.split()
                    if "tagged" in words and words.index("tagged") + 1 < len(words):
                        priority = words[words.index("tagged") + 1].lower()
                        entities["priority"] = "high" if "urgent" in priority else priority
            else:
                intent = "tasks_general"
        elif "where" in query_lower or "find note" in query_lower:
            intent = "find_notes"
            # Extract the keyword (e.g., "marketing budget")
            keyword_match = re.search(
                r"(?:where.*write about|find note.*related to)\s+(.+)", query_lower
            )
            if keyword_match:
                entities["keyword"] = keyword_match.group(1).strip("'\"")
            else:
                entities["keyword"] = (
                    query_lower.split("find notes related to")[-1].strip()
                    if "find notes" in query_lower
                    else query_lower.split("where")[-1].strip()
                )
        elif "how much" in query_lower:
            intent = "specific_info"
            # Extract the specific item (e.g., "strawberry" in "how much strawberry needed")
            item_match = re.search(r"how much\s+([a-z\s]+)\s+needed", query_lower)
            if item_match:
                entities["specific_info"] = item_match.group(1).strip()
        elif "audio" in query_lower or "recording" in query_lower:
            intent = "find_audio"
            # Extract the keyword (e.g., "quarterly report")
            keyword_match = re.search(
                r"(?:mentioned|about)\s+['\"]([^'\"]+)['\"]", query_lower
            )
            if keyword_match:
                entities["keyword"] = keyword_match.group(1)
            else:
                words = query_lower.split()
                if "mentioned" in words and words.index("mentioned") + 1 < len(words):
                    entities["keyword"] = words[words.index("mentioned") + 1]
        elif "key takeaway" in query_lower or "meeting on" in query_lower:
            intent = "meeting_takeaway"
            # Extract the date (e.g., "October 26th")
            date_match = re.search(r"meeting on\s+(.+)", query_lower)
            if date_match:
                entities["date"] = date_match.group(1)
        else:
            intent = "qa"  # Default to question-answering (e.g., "What is OSI?")

        # Step 3: Handle the intent
        answer = ""
        related_elements = []

        if intent == "tasks_due" or intent == "tasks_general":
            # Filter tasks
            tasks = [el for el in user_elements if el.get("type") == "task"]
            if not tasks:
                answer = "You have no tasks."
            else:
                if "upcoming" in query_lower or intent == "tasks_due":
                    # Calculate date range (e.g., next 14 days or next week)
                    today = datetime.now(timezone.utc)
                    if "next week" in query_lower:
                        start_date = today + timedelta(days=(7 - today.weekday()))
                        end_date = start_date + timedelta(days=6)
                    else:
                        start_date = today
                        end_date = today + timedelta(days=14)

                    filtered_tasks = []
                    for task in tasks:
                        due_date_str = task.get("due_date")
                        if due_date_str:
                            try:
                                due_date = datetime.fromisoformat(
                                    due_date_str.replace("Z", "+00:00")
                                )
                                if start_date <= due_date <= end_date:
                                    filtered_tasks.append(task)
                            except ValueError:
                                continue

                    if not filtered_tasks:
                        answer = f"You have no tasks due between {start_date.strftime('%B %d, %Y')} and {end_date.strftime('%B %d, %Y')}."
                    else:
                        task_list = [
                            f"- \"{task['title']}\" due on {task['due_date']}"
                            for task in filtered_tasks
                        ]
                        answer = (
                            f"Your tasks due between {start_date.strftime('%B %d, %Y')} and {end_date.strftime('%B %d, %Y')} are:\n"
                            + "\n".join(task_list)
                        )
                        related_elements = [
                            element_to_dict(task) for task in filtered_tasks
                        ]
                else:
                    answer = "Here are your tasks:\n" + "\n".join(
                        [
                            f"- \"{task['title']}\" due on {task.get('due_date', 'N/A')}"
                            for task in tasks
                        ]
                    )
                    related_elements = [element_to_dict(task) for task in tasks]

        elif intent == "tasks_priority":
            priority = entities["priority"]
            if not priority:
                answer = "Please specify a priority to search for (e.g., 'urgent', 'high', 'medium', 'low')."
            else:
                tasks = [
                    el
                    for el in user_elements
                    if el.get("type") == "task"
                    and el.get("priority", "").lower() == priority
                ]
                if not tasks:
                    answer = f"You have no tasks with priority '{priority}'."
                else:
                    task_list = [
                        f"- \"{task['title']}\" due on {task.get('due_date', 'N/A')}"
                        for task in tasks
                    ]
                    answer = f"The tasks with priority '{priority}' are:\n" + "\n".join(task_list)
                    related_elements = [element_to_dict(task) for task in tasks]

        elif intent == "find_notes":
            keyword = entities["keyword"]
            if not keyword:
                answer = "Please specify a keyword to search for in notes."
            else:
                notes = [el for el in user_elements if el.get("type") == "note"]
                matching_notes = [
                    note
                    for note in notes
                    if (note.get("title") and keyword.lower() in note["title"].lower())
                    or (
                        note.get("content")
                        and keyword.lower() in note["content"].lower()
                    )
                ]
                if not matching_notes:
                    answer = f"I couldn’t find any notes related to '{keyword}'."
                else:
                    note_list = []
                    for note in matching_notes:
                        snippet = (
                            note["content"][:100] + "..."
                            if note.get("content") and len(note["content"]) > 100
                            else note.get("content", "")
                        )
                        note_list.append(
                            f"- \"{note['title']}\" created on {note['created_at']}: \"{snippet}\""
                        )
                    answer = f"The notes related to '{keyword}' are:\n" + "\n".join(
                        note_list
                    )
                    related_elements = [
                        element_to_dict(note) for note in matching_notes
                    ]

        elif intent == "specific_info":
            item = entities["specific_info"]
            if not item:
                answer = "Please specify an item to search for (e.g., 'how much strawberry needed')."
            else:
                notes = [el for el in user_elements if el.get("type") == "note"]
                matching_notes = [
                    note
                    for note in notes
                    if (note.get("title") and item.lower() in note["title"].lower())
                    or (note.get("content") and item.lower() in note["content"].lower())
                ]
                if not matching_notes:
                    answer = f"I couldn’t find any notes mentioning '{item}'."
                else:
                    # Look for a pattern like "X cups of item"
                    for note in matching_notes:
                        content = note.get("content", "")
                        pattern = rf"(\d+\.?\d*)\s*(cup?|tablespoons?|tsp|teaspoons?)\s*of\s*{re.escape(item)}"
                        match = re.search(pattern, content, re.IGNORECASE)
                        if match:
                            amount = match.group(1)
                            unit = match.group(2)
                            answer = f"You need {amount} {unit} of {item}, as mentioned in your note titled \"{note['title']}\"."
                            related_elements = [element_to_dict(note)]
                            break
                    else:
                        answer = f"I found notes mentioning '{item}', but couldn’t determine the quantity."
                        related_elements = [
                            element_to_dict(note) for note in matching_notes
                        ]

        elif intent == "find_audio":
            keyword = entities["keyword"]
            if not keyword:
                answer = "Please specify a keyword to search for in audio recordings."
            else:
                audios = [el for el in user_elements if el.get("type") == "audio"]
                matching_audios = [
                    audio
                    for audio in audios
                    if audio.get("transcription")
                    and keyword.lower() in audio["transcription"].lower()
                ]
                if not matching_audios:
                    answer = (
                        f"I couldn’t find an audio recording mentioning '{keyword}'."
                    )
                else:
                    audio_list = []
                    for audio in matching_audios:
                        transcription = audio["transcription"]
                        snippet = (
                            transcription[:100] + "..."
                            if len(transcription) > 100
                            else transcription
                        )
                        audio_list.append(
                            f"- \"{audio['title']}\" recorded on {audio['created_at']}: \"{snippet}\""
                        )
                    answer = (
                        f"The audio recordings mentioning '{keyword}' are:\n"
                        + "\n".join(audio_list)
                    )
                    related_elements = [
                        element_to_dict(audio) for audio in matching_audios
                    ]

        elif intent == "meeting_takeaway":
            date_str = entities["date"]
            if not date_str:
                answer = "Please specify a date for the meeting (e.g., 'meeting on October 26th')."
            else:
                try:
                    meeting_date = parse_date(date_str, fuzzy=True)
                    meeting_date_str = meeting_date.strftime("%Y-%m-%d")
                    notes = [el for el in user_elements if el.get("type") == "note"]
                    matching_notes = [
                        note
                        for note in notes
                        if note.get("created_at", "").startswith(meeting_date_str)
                        or (note.get("title") and meeting_date_str in note["title"])
                    ]
                    if not matching_notes:
                        answer = (
                            f"I couldn’t find a note about a meeting on {date_str}."
                        )
                    else:
                        note = matching_notes[0]  # Take the first matching note
                        content = note.get("content", "")
                        # Extract the first sentence as the key takeaway
                        sentences = content.split(". ")
                        takeaway = (
                            sentences[0] + "."
                            if sentences
                            else "No detailed content available."
                        )
                        answer = f"The key takeaway from the meeting on {date_str}, found in a note titled \"{note['title']}\", is: \"{takeaway}\""
                        related_elements = [element_to_dict(note)]
                except ValueError:
                    answer = f"I couldn’t parse the date '{date_str}'. Please use a format like 'October 26th'."

        elif intent == "qa":
            # Filter relevant notes first
            notes = [el for el in user_elements if el.get("type") == "note"]
            matching_notes = [
                note
                for note in notes
                if (note.get("title") and query_lower in note["title"].lower())
                or (note.get("content") and query_lower in note["content"].lower())
            ]
            if not matching_notes:
                answer = (
                    f"I couldn’t find any information about '{query}' in your notes."
                )
            else:
                # Combine content from matching notes
                context = " ".join(
                    [
                        f"{note.get('title', '')}: {note.get('content', '')}"
                        for note in matching_notes
                        if note.get("content")
                    ]
                )
                if not context.strip():
                    answer = "I found some relevant notes, but they don’t contain enough content to answer your question."
                else:
                    # Use Hugging Face QA pipeline
                    result = qa_pipeline({"question": query, "context": context})
                    answer_text = result["answer"]
                    score = result["score"]

                    if score < 0.1:
                        # Fallback: Summarize the first few sentences
                        first_note = matching_notes[0]
                        content = first_note.get("content", "")
                        sentences = content.split(". ")
                        summary = (
                            " ".join(sentences[:2]) + "."
                            if len(sentences) >= 2
                            else content
                        )
                        answer = f"{summary} (From your note titled \"{first_note['title']}\".)"
                    else:
                        answer = f"{answer_text} (From your note titled \"{matching_notes[0]['title']}\".)"

                related_elements = [element_to_dict(note) for note in matching_notes]

        return JSONResponse(
            content={"answer": answer, "elements": related_elements},
            headers={"Access-Control-Allow-Origin": "http://localhost:5173"},
        )
    except Exception as e:
        print(f"Error in /smart_search endpoint: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse(
            content={
                "answer": "An error occurred while processing your query.",
                "elements": [],
            },
            status_code=500,
            headers={"Access-Control-Allow-Origin": "http://localhost:5173"},
        )