from pymongo import MongoClient

MONGO_URI = "mongodb+srv://orbituser:OrbitPass123!@orbit-cluster.4bi15tv.mongodb.net/?retryWrites=true&w=majority&appName=orbit-cluster"
client = MongoClient(MONGO_URI)
db = client["orbit"]
collection = db["canvas_elements"]

# Remove tags field from tasks
collection.update_many(
    {"type": "task"},
    {"$unset": {"tags": ""}}
)

# Add transcription field to audio elements
collection.update_many(
    {"type": "audio", "transcription": {"$exists": False}},
    {"$set": {"transcription": None}}
)

print("Database updated: removed tags field from tasks and added transcription field to audio elements.")