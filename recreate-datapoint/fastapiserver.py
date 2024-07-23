from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import json

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    with open("static/index.html", "r") as file:
        content = file.read()
    return HTMLResponse(content=content)

@app.get("/getFirst")
async def get_first_entry():
    with open("full_raw_paperclip.ndjson", "r") as file:
        firstLine = file.readline()
        if firstLine:
            return json.loads(firstLine)
        else:
            return {"error": "File is empty"}