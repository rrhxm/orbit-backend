from pymongo import MongoClient

MONGO_URI = "mongodb+srv://orbituser:OrbitPass123!@orbit-cluster.4bi15tv.mongodb.net/?retryWrites=true&w=majority&appName=orbit-cluster"
client = MongoClient(MONGO_URI)
db = client.get_database()

def get_database():
    return db