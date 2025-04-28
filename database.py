from pymongo import MongoClient

MONGO_URI = "mongodb+srv://orbituser:OrbitPass123!@orbit-cluster.4bi15tv.mongodb.net/?retryWrites=true&w=majority&appName=orbit-cluster"
client = MongoClient(MONGO_URI)
db = client.get_database()

# Ensure index on user_id for faster queries
def ensure_indexes():
    db["canvas_elements"].create_index("user_id")

def get_database():
    ensure_indexes()  # Ensure indexes are created on startup
    return db