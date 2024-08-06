import os
import logging
import asyncio
import redis.asyncio as redis

from fastapi import (
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    WebSocketException,
    status,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import json
import uuid

class GameData(BaseModel):
    game_id: str | None = None
    player_name: str

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

redis_client = None
try:
    redis_client = redis.Redis(host=os.getenv("REDIS_HOST") or "localhost",
                               port=int(os.getenv("REDIS_PORT") or 6379),
                               decode_responses=True)
    logging.info("Redis connection established")
except Exception as e:
    logging.error(f"Failed to connect to Redis: {e}")

def is_valid_uuid(uuid_string):
    try:
        val = uuid.UUID(uuid_string, version=4)
    except ValueError:
        return False
    return str(val) == uuid_string

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    if not redis_client:
        logging.error("game: Redis is not available")
        raise WebSocketException(code=status.WS_1011_INTERNAL_ERROR)
    game_id = websocket.cookies.get('quickdrawvs_game_id')
    player_id = websocket.cookies.get('quickdrawvs_player_id')
    if game_id is None or player_id is None:
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION,
                                 reason="No game information cookie")
    elif not is_valid_uuid(game_id) or not is_valid_uuid(player_id):
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION,
                                 reason="Invalid game information cookie")
    game_data = await redis_client.get(f"game:{game_id}")
    if not game_data:
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)
    players = await redis_client.lrange(f"game:{game_id}:players", 0, -1) # type: ignore
    if player_id not in players: # type: ignore
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)
    await websocket.accept()
    logging.info("Websocket connection accepted.")
    try:
        game_data = json.loads(await redis_client.get(f"game:{game_id}")) # type: ignore
        if game_data["status"] == "waiting":
            nplayers = await redis_client.llen(f"game:{game_id}:players") # type: ignore
            while nplayers < 2: # type: ignore
                await asyncio.sleep(1)
                nplayers = await redis_client.llen(f"game:{game_id}:players") # type: ignore
            logging.info(f"More than 2 players found, starting")
            game_data["status"] = "playing"
            redis_client.set(f"game:{game_id}", json.dumps(game_data))
        else:
            raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION,
                                     reason="Game has already started")
        # Game has started
        async for data in websocket.iter_json():
            # Send to ML model
            pass
        await websocket.close()
    except WebSocketDisconnect:
        await redis_client.lrem(f"game:{game_id}:players", -1, player_id) # type: ignore

@app.post("/create-game")
async def create_game(data: GameData, request: Request):
    if not redis_client:
        logging.error("create_game: Redis is not available")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Internal database error")
    try:
        game_id = data.game_id
        if game_id is None or await redis_client.get(f"game:{game_id}") is None:
            game_id = str(uuid.uuid4())
            game_data = {
                "status": "waiting",
            }
            await redis_client.set(f"game:{game_id}", json.dumps(game_data))
            logging.info(f"Created new game with ID: {game_id}")

        player_id = str(uuid.uuid4())
        player_data = {
            "name": data.player_name,
            "id": player_id,
        }
        await redis_client.rpush(f"game:{game_id}:players", json.dumps(player_data)) # type: ignore
        logging.info(f"Created new host player")
        return {"game_id": game_id, "player_id": player_id}
    except Exception as e:
        logging.error(f"create_game: Error creating game: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

# These have to come after or the endpoints are overriden and incorrectly
# routed.
NEXT_DIR = "/quickdraw/out"
app.mount("/", StaticFiles(directory=NEXT_DIR, html = True), name="static")
