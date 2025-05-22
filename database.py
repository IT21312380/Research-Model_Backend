from pymongo import MongoClient

def get_database():
    try:
        client = MongoClient(
            "mongodb+srv://intothejungle:Kjhgfdsa08#@intothejungle.vfm0u.mongodb.net/?retryWrites=true&w=majority&appName=IntoTheJungle"
        )
        db = client["IntothejungleDB"]  # Replace with your database name
        collection = db["animal_predictions"]  # Replace with your collection name
        print("Connected to MongoDB successfully!")
        return db, collection
    except Exception as e:
        raise RuntimeError(f"Error connecting to MongoDB: {str(e)}")
