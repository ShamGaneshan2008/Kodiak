from fastapi import FastAPI

app = FastAPI(title="Kodiak")

@app.get("/health")
async def health():
    return {"status": "healthy"}