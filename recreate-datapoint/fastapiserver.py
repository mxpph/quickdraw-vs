from fastapi import FastAPI
import json

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/getFirst")
async def get_first_entry():
    with open("full_raw_paperclip.ndjson", "r") as file:
        firstLine = file.readline()
        if firstLine:
            return json.loads(firstLine)
        else:
            return {"error": "File is empty"}