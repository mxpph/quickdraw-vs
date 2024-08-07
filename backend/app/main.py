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

async def pubsub_single(pubsub: redis.client.PubSub, channel: str):
    try:
        await pubsub.subscribe(channel)
        while True:
            message = await pubsub.get_message()
            if message and message["type"] == "message":
                break
            await asyncio.sleep(0.001) # be nice to the system :)

        await pubsub.unsubscribe(channel)
    except redis.PubSubError as e:
        logging.error(f"Exception in pubsub_single: {e}")

async def pubsub_loop(websocket: WebSocket, pubsub: redis.client.PubSub):
    try:
        while True:
            message = await pubsub.get_message()
            if message and message["type"] == "message":
                # Broadcast back to all clients
                await websocket.send_json(message["data"])
            await asyncio.sleep(0.001) # be nice to the system :)

    except redis.PubSubError as e:
        logging.error(f"Exception in pubsub_loop: {e}")

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
    game_data = json.loads(game_data) if game_data else None
    if not game_data or game_data["status"] != "waiting":
        logging.debug("ws: Invalid game")
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)

    player_ids = await redis_client.lrange(f"game:{game_id}:players", 0, -1)
    if player_id not in player_ids:
        logging.debug("ws: Not a player in this game ID")
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)

    # All is okay, so accept the connection.
    await websocket.accept()
    logging.info(f"Websocket connection accepted for {player_id}")

    async def websocket_loop():
        nonlocal game_id, player_id, websocket
        while True:
            # Cannot use iter_json here as it ignores WebSocketDisconnect.
            data = await websocket.receive_json()
            # TODO: add the necessary messages for gameplay (new word, game end)
            match data.get("type"):
                case "test":
                    await redis_client.publish(f"game:{game_id}:channel",
                                            json.dumps({"message": f"{player_id} says hi!"}))
                case _:
                    raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)

    try:
        nplayers = await redis_client.llen(f"game:{game_id}:players")
        async with redis_client.pubsub(ignore_subscribe_messages=True) as pubsub:
            # Wait for the right amount of players before starting.
            # TODO: Make it so the host starts the game instead.
            if nplayers < game_data["max_players"]:
                await pubsub_single(pubsub, f"game:{game_id}:start")
            else:
                logging.debug(f"Found enough players for {game_id}")
                game_data["status"] = "playing"
                await redis_client.set(f"game:{game_id}", json.dumps(game_data))
                await redis_client.publish(f"game:{game_id}:start", "start")

            await pubsub.subscribe(f"game:{game_id}:channel")
            await asyncio.gather(
                websocket_loop(),
                pubsub_loop(websocket, pubsub),
            )

    except WebSocketDisconnect:
        logging.info(f"Client {player_id} dropped")
        await redis_client.lrem(f"game:{game_id}:players", -1, player_id)
        if await redis_client.llen(f"game:{game_id}:players") == 0:
            await redis_client.delete(f"game:{game_id}:players")
            await redis_client.delete(f"game:{game_id}")

    except redis.PubSubError as e:
        logging.error(f"Pubsub exception in websocket_endpoint: {e}")
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)

@app.post("/create-game")
async def create_game(data: GameData, request: Request):
    if not redis_client:
        logging.error("create_game: Redis is not available")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Internal database error")
    try:
        game_id = data.game_id
        is_host = False
        if (    game_id is None or not is_valid_uuid(game_id)
                or await redis_client.get(f"game:{game_id}") is None):
            is_host = True
            game_id = str(uuid.uuid4())
            game_data = {
                "status": "waiting",
                "current_round": 1,
                "max_players": 2, # TODO: These should be specified in `data` (and validated)
                "rounds": 5,
            }
            await redis_client.set(f"game:{game_id}", json.dumps(game_data))
            logging.info(f"Created new game with ID: {game_id}")

        player_id = str(uuid.uuid4())
        player_data = {
            "name": data.player_name,
            "is_host": is_host,
        }
        await redis_client.rpush(f"game:{game_id}:players", player_id)
        await redis_client.set(f"game:{game_id}:players:{player_id}", json.dumps(player_data))
        logging.info(f"Created new player with ID: {player_id}")
        return {"game_id": game_id, "player_id": player_id}
    except Exception as e:
        logging.error(f"create_game: Error creating game: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

# These have to come after or the endpoints are overriden and incorrectly
# routed.
NEXT_DIR = "/quickdraw/out"
app.mount("/", StaticFiles(directory=NEXT_DIR, html = True), name="static")
