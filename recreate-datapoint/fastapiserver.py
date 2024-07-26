from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import json

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

paperclipData = open("full_raw_paperclip.ndjson", "r")

@app.get("/")
async def root():
    with open("static/index.html", "r") as file:
        content = file.read()
    return HTMLResponse(content=content)

@app.get("/get")
async def get_first_entry():
        line = paperclipData.readline()
        if line:
            return json.loads(line)
        else:
            return {"error": "File is empty"}