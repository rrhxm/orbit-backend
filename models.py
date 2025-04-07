from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, timezone

# Common Base Model
class BaseElement(BaseModel):
    x: int = Field(..., description="X position on canvas")
    y: int = Field(..., description="Y position on canvas")
    type: str = Field(..., description="Type of element (e.g., note, task, image, audio)")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Date & time created")
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Date & time updated")

# Note Model
class Note(BaseElement):
    type: str = "note"
    title: str = Field(..., description="Title of the note")
    content: str = Field(..., description="Content of the note")

# Task Model
class Task(BaseElement):
    type: str = "task"
    title: str = Field(..., description="Title of the task")
    due_date: str = Field(..., description="Due date of the task (YYYY-MM-DD)")
    due_time: Optional[str] = Field(None, description="Due time of the task (HH:MM)")
    priority: str = Field(..., description="Task priority: low, medium, high")
    repeat: str = Field("no", description="Repeat daily: yes or no")
    completed: bool = Field(False, description="Task completion status")
    last_reset: str = Field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d"), description="Last reset date for repeating tasks")
    is_edited: bool = Field(False, description="Whether the task has been edited")

# Image Model
class Image(BaseElement):
    type: str = "image"
    title: str = Field(..., description="Title of the image")
    image_data: Optional[str] = Field(None, description="Base64-encoded image data")

# Audio Model
class Audio(BaseElement):
    type: str = "audio"
    title: str = Field(..., description="Title of the audio")
    audio_data: Optional[str] = Field(None, description="Base64-encoded audio data")
    transcription: Optional[str] = Field(None, description="Transcription of the audio content")