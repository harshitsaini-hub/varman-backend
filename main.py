from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "AMOR backend running"}

@app.post("/api/test")
def test_endpoint(data: dict):
    print("Received:", data)
    return {"status": "received", "data": data}