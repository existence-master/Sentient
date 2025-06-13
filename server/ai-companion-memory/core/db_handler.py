import pymongo
from .config import MONGO_URI, DB_NAME

class DBHandler:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DBHandler, cls).__new__(cls)
            cls._instance.client = pymongo.MongoClient(MONGO_URI)
            cls._instance.db = cls._instance.client[DB_NAME]
            print("MongoDB connection established.")
            cls._instance._ensure_indexes()
        return cls._instance

    def _ensure_indexes(self):
        # TTL index for short-term memories
        self.db.short_term_memories.create_index("expires_at", expireAfterSeconds=0)
        # Standard indexes for faster lookups
        self.db.long_term_memories.create_index([("user_id", 1)])
        self.db.short_term_memories.create_index([("user_id", 1)])
        self.db.contacts.create_index([("user_id", 1)])
        self.db.contacts.create_index([("user_id", 1), ("name", 1)])
        print("Database indexes ensured.")

    def get_collection(self, collection_name: str):
        return self.db[collection_name]

    def execute_query(self, collection_name: str, operation: str, query: dict):
        collection = self.get_collection(collection_name)
        if not hasattr(collection, operation):
            raise ValueError(f"Invalid MongoDB operation: {operation}")
        
        # The query dict should already be validated and formed correctly
        # e.g., for find: query={'filter': {...}, 'projection': {...}}
        # e.g., for insert_one: query={'document': {...}}
        
        method = getattr(collection, operation)
        result = method(**query)

        # Make results JSON serializable
        if isinstance(result, pymongo.results.InsertOneResult):
            return {"inserted_id": str(result.inserted_id)}
        if isinstance(result, pymongo.results.UpdateResult):
            return {"matched_count": result.matched_count, "modified_count": result.modified_count}
        if isinstance(result, pymongo.cursor.Cursor):
            return list(result) # Be careful with large result sets
        
        return result


# Singleton instance
db_handler = DBHandler()
