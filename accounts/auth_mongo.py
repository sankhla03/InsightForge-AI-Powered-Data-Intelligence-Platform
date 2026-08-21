from pymongo import MongoClient
from django.conf import settings
from werkzeug.security import generate_password_hash, check_password_hash

def get_mongo_db():
    # Bug H2 fix: reduce timeout from 3000ms to 1500ms to halve the delay
    # when MongoDB is not running and the app falls back to Django auth.
    client = MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=1500)
    return client[settings.MONGO_DB_NAME]


def create_user(username, email, password):
    db = get_mongo_db()
    users = db.users

    if users.find_one({"email": email}):
        return False, "User already exists"

    users.insert_one({
        "username": username,
        "email": email,
        "password": generate_password_hash(password)
    })
    return True, "User created"


def authenticate_user(email, password):
    db = get_mongo_db()
    users = db.users

    user = users.find_one({"email": email})
    if not user:
        return None

    if check_password_hash(user["password"], password):
        return user

    return None