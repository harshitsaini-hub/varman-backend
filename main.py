from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "AMOR backend running"}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    print("📸 File received:", file.filename, flush=True)

    contents = await file.read()
    print("File size:", len(contents), flush=True)

    return {"filename": file.filename, "size": len(contents)}