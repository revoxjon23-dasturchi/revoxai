import os
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="REVOX AI")

API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]


@app.get("/")
def home():
    return {"app": "REVOX AI", "status": "running"}


@app.get("/health")
def health():
    return {"status": "ok", "app": "REVOX AI"}


@app.post("/chat")
async def chat(request: ChatRequest):
    if not API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Serverda Gemini API kaliti sozlanmagan"
        )

    contents = []

    for message in request.messages[-30:]:
        role = "model" if message.role == "assistant" else "user"
        content = message.content.strip()

        if content:
            contents.append({
                "role": role,
                "parts": [{"text": content}]
            })

    if not contents or contents[-1]["role"] != "user":
        raise HTTPException(
            status_code=400,
            detail="Oxirgi xabar foydalanuvchidan bo'lishi kerak"
        )

    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{MODEL}:generateContent"
    )

    payload = {
        "systemInstruction": {
            "parts": [{
                "text": (
                    "You are REVOX AI, a helpful AI assistant. "
                    "Reply naturally in the language used by the user. "
                    "Be clear, useful, and honest."
                )
            }]
        },
        "contents": contents,
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 2048
        }
    }

    try:
        async with httpx.AsyncClient(timeout=90) as client:
            response = await client.post(
                url,
                headers={
                    "x-goog-api-key": API_KEY,
                    "Content-Type": "application/json"
                },
                json=payload
            )

        if response.status_code != 200:
            raise HTTPException(
                status_code=502,
                detail=f"Gemini API xatosi: {response.text[:400]}"
            )

        data = response.json()
        candidates = data.get("candidates", [])

        if not candidates:
            raise HTTPException(
                status_code=502,
                detail="Gemini javob qaytarmadi"
            )

        parts = candidates[0].get("content", {}).get("parts", [])
        answer = "\n".join(
            part["text"] for part in parts if "text" in part
        ).strip()

        if not answer:
            raise HTTPException(
                status_code=502,
                detail="AI javobi bo'sh qaytdi"
            )

        return {"reply": answer}

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="AI javobi uzoq kutilmoqda. Qayta urinib ko'ring."
        )
    except httpx.HTTPError:
        raise HTTPException(
            status_code=502,
            detail="Gemini serveriga ulanib bo'lmadi"
        )
