from fastapi import FastAPI

app = FastAPI(title="Vocabulary Learning API")


@app.get("/api")
def home():
    return {"message": "Vocabulary Learning API is running"}