import os
import sys
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

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

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

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_single(self, data: str, websocket: WebSocket):
        await websocket.send_json(data)

    async def broadcast(self, data: str):
        for connection in self.active_connections:
            await connection.send_json(data)

manager = ConnectionManager()

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

    # Check the validity of the incoming connection
    game_id = websocket.cookies.get('quickdrawvs_game_id')
    player_id = websocket.cookies.get('quickdrawvs_player_id')

    if game_id is None or player_id is None:
        logging.debug("ws: No game information cookie")
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)

    elif not is_valid_uuid(game_id) or not is_valid_uuid(player_id):
        logging.debug("ws: Invalid game information cookie")
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)

    game_data = await redis_client.get(f"game:{game_id}")
    if not game_data or json.loads(game_data)["status"] != "waiting":
        logging.debug("ws: Invalid game")
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)

    players_datas = await redis_client.lrange(f"game:{game_id}:players", 0, -1)
    player_ids = [ json.loads(player)["id"] for player in players_datas ]
    if player_id not in player_ids:
        logging.debug("ws: Not a player in this game ID")
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)

    # All is okay, so accept the connection.
    await manager.connect(websocket)
    logging.info(f"Websocket connection accepted for {player_id}")
    try:
        game_data = json.loads(await redis_client.get(f"game:{game_id}"))
        logging.debug(game_data)
        if game_data["status"] == "waiting":
            nplayers = await redis_client.llen(f"game:{game_id}:players")
            while nplayers < 2:
                await asyncio.sleep(1)
                nplayers = await redis_client.llen(f"game:{game_id}:players")
            logging.info(f"More than 2 players found, starting")
            game_data["status"] = "playing"
            await redis_client.set(f"game:{game_id}", json.dumps(game_data))
        else:
            raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)
        # Game has started
        async for data in websocket.iter_json():
            # Send to ML model
            pass
        await websocket.close()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await redis_client.lrem(f"game:{game_id}:players", -1, player_id)

@app.post("/create-game")
async def create_game(data: GameData, request: Request):
    if not redis_client:
        logging.error("create_game: Redis is not available")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Internal database error")
    try:
        game_id = data.game_id
        if (    game_id is None or not is_valid_uuid(game_id)
                or await redis_client.get(f"game:{game_id}") is None):
            # TODO: Make player a host here
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
        await redis_client.rpush(f"game:{game_id}:players", json.dumps(player_data))
        logging.info(f"Created new host player with ID: {player_id}")
        return {"game_id": game_id, "player_id": player_id}
    except Exception as e:
        logging.error(f"create_game: Error creating game: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

# These have to come after or the endpoints are overriden and incorrectly
# routed.
NEXT_DIR = "/quickdraw/out"
app.mount("/", StaticFiles(directory=NEXT_DIR, html = True), name="static")
