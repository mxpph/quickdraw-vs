import os
import sys
import logging
import asyncio
import json
import uuid

import redis.asyncio as redis
from fastapi import (
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    WebSocketException,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.util import util

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class CreateGameData(BaseModel):
    """Required data for create-game HTTP request"""
    player_name: str
    max_players: str
    rounds: str

class JoinGameData(BaseModel):
    """Required data for join-game HTTP request"""
    game_id: str
    player_name: str

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://quickdraw-vs.com",
    "http://www.quickdraw-vs.com",
    "app"
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
except redis.ConnectionError as e:
    logging.error("Failed to connect to Redis: %s", e)
    sys.exit(1)

async def pubsub_single(pubsub: redis.client.PubSub, channel: str):
    """
    Wait for a single pubsub message on `channel`, then return.
    """
    try:
        await pubsub.subscribe(channel)
        while True:
            message = await pubsub.get_message()
            if message and message["type"] == "message":
                await pubsub.unsubscribe(channel)
                return json.loads(message["data"])["type"]
            await asyncio.sleep(0.001) # be nice to the system :)

    except redis.PubSubError as e:
        logging.error("Exception in pubsub_single: %s", e)

async def receive_start_game(websocket: WebSocket, game_id: str, game_data: dict):
    """
    Wait for the start game message from `websocket`, then update the status
    in the database.
    Throws an exception if any other message is recevied.
    """
    data = await websocket.receive_json()
    if data.get("type") != "start_game":
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION,
                                 reason="Unknown message received")
    if await redis_client.llen(f"game:{game_id}:players") < 2:
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION,
                                 reason="Not enough players to start the game")
    logging.debug("Got start game message for %s", game_id)
    await redis_client.publish(f"game:{game_id}:start", '{"type": "start"}')
    game_data["status"] = "playing"
    await redis_client.set(f"game:{game_id}", json.dumps(game_data))

async def get_scoreboard(game_id: str) -> list[tuple[str, int]]:
    """Gets the scoreboard of points for each player of the game"""
    wins = await redis_client.lrange(f"game:{game_id}:wins", 0, -1)
    return util.most_common(wins)

async def send_next_round(game_id: str, current_round: int):
    """Send the next round message to all clients"""
    word = util.get_random_word()
    await redis_client.publish(f"game:{game_id}:channel",
                               json.dumps({"type": "next_round",
                                           "round": current_round + 1,
                                           "word": word}))

async def send_game_over(game_id: str, scoreboard: list[tuple[str, int]]):
    """Send the game over message to all clients"""
    await redis_client.publish(f"game:{game_id}:channel",
                               json.dumps({"type": "game_over",
                                           "scoreboard": scoreboard}))

async def cancel_game(game_id: str):
    await redis_client.publish(f"game:{game_id}:start",
                         json.dumps({"type": "cancel"}))
    await redis_client.delete(f"game:{game_id}:players")
    await redis_client.delete(f"game:{game_id}")

async def pubsub_loop(websocket: WebSocket, pubsub: redis.client.PubSub):
    """Send back messages received on Pub/sub via the WebSocket"""
    try:
        while True:
            message = await pubsub.get_message()
            if message and message["type"] == "message":
                # Broadcast back to all clients
                await websocket.send_text(message["data"])
                if json.loads(message["data"])["type"] == "game_over":
                    await websocket.close()
                    raise WebSocketDisconnect()
            await asyncio.sleep(0.001) # be nice to the system :)

    except redis.PubSubError as e:
        logging.error("Exception in pubsub_loop: %s", e)

async def websocket_loop(
    websocket: WebSocket,
    game_id: str,
    player_id: str,
    game_data: dict,
    player_data: dict
):
    """Handle data send by clients through websocket connection"""
    while True:
        # Cannot use iter_json here as it ignores WebSocketDisconnect.
        data = await websocket.receive_json()
        match data.get("type"):
            case "win":
                logging.debug("Got win message from \'%s\'", player_data['name'])
                round_no = await redis_client.rpush(f"game:{game_id}:wins", player_data['name'])
                if round_no == game_data["rounds"]:
                    scoreboard = await get_scoreboard(game_id)
                    await send_game_over(game_id, scoreboard)
                    return
                await send_next_round(game_id, round_no)
            case _:
                raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION,
                                         reason="Unknown message received")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle incoming websocket connections"""
    if not redis_client:
        logging.error("game: Redis is not available")
        raise WebSocketException(code=status.WS_1011_INTERNAL_ERROR)

    # Check the validity of the incoming connection
    game_id = websocket.cookies.get('quickdrawvs_game_id')
    player_id = websocket.cookies.get('quickdrawvs_player_id')

    if game_id is None or player_id is None:
        logging.debug("ws: No game information cookie")
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)

    if not util.is_valid_uuid(game_id) or not util.is_valid_uuid(player_id):
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
    player_data = json.loads(await redis_client.get(f"game:{game_id}:players:{player_id}"))

    # All is okay, so accept the connection.
    await websocket.accept()
    logging.info("Websocket connection accepted for %s", player_id)

    try:
        async with redis_client.pubsub(ignore_subscribe_messages=True) as pubsub:
            # Wait for host to start the game
            if player_data["is_host"]:
                await receive_start_game(websocket, game_id, game_data)
            elif await pubsub_single(pubsub, f"game:{game_id}:start") == "cancel":
                await websocket.send_text('{"type": "cancel"}')
                await websocket.close()
                raise WebSocketDisconnect()

            await pubsub.subscribe(f"game:{game_id}:channel")
            if player_data["is_host"]:
                await send_next_round(game_id, 0)
            await asyncio.gather(
                websocket_loop(websocket, game_id, player_id, game_data, player_data),
                pubsub_loop(websocket, pubsub),
            )

    except WebSocketDisconnect:
        logging.debug("Client %s dropped", player_data['name'])
        await util.random_sleep(0.1) # sleep for up to 0.1s to stagger many requests at once
        # Delete player details
        await redis_client.delete(f"game:{game_id}:players:{player_id}")
        game_data = await redis_client.get(f"game:{game_id}")
        game_status = json.loads(game_data)["status"] if game_data else None
        # If the host disconnects before the game starts, we just cancel
        # the game.
        if player_data["is_host"] and game_status == "waiting":
            await cancel_game(game_id)
            return
        await redis_client.lrem(f"game:{game_id}:players", -1, player_id)
        # Delete game_id if the lobby is now empty
        if await redis_client.llen(f"game:{game_id}:players") == 0:
            logging.debug("Game %s is empty, deleting", game_id)
            await redis_client.delete(f"game:{game_id}:players")
            await redis_client.delete(f"game:{game_id}")

    except redis.PubSubError as e:
        logging.error("Pubsub exception in websocket_endpoint: %s", e)
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR,
                              reason="Internal server error, try again later")

MIN_PLAYERS = 2
MAX_PLAYERS = 6
MIN_ROUNDS = 2
MAX_ROUNDS = 10
MAX_LEN_PLAYER_NAME = 16

async def check_too_many_players(game_id: str) -> bool:
    """
    Return true if adding one more player would surpass the maximum for
    a game.
    """
    length = await redis_client.llen(f"game:{game_id}:players")
    maximum = int(json.loads(await redis_client.get(f"game:{game_id}"))["max_players"])
    return length + 1 > maximum

async def insert_player(game_id: str, player_id: str, player_data: dict):
    """Inserts a player into a game"""
    await redis_client.rpush(f"game:{game_id}:players", player_id)
    await redis_client.set(f"game:{game_id}:players:{player_id}", json.dumps(player_data))
    logging.info("Created new player with ID: %s", player_id)

@app.post("/join-game")
async def join_game(data: JoinGameData):
    """Handle a POST request to join an existing game"""
    if not redis_client:
        logging.error("create_game: Redis is not available")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Internal database error")
    try:
        game_id = data.game_id
        if (    game_id is None or game_id == "" or not util.is_valid_uuid(game_id)
                or await redis_client.get(f"game:{game_id}") is None):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                                detail="Invalid game ID")
        if await check_too_many_players(game_id):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                                detail="This game is full")
        player_id = str(uuid.uuid4())
        player_data = {
            "name": data.player_name,
            "is_host": False,
        }
        await insert_player(game_id, player_id, player_data)
        return {"game_id": game_id,
                "player_id": player_id,
                "is_host": 'False'}
    except HTTPException:
        raise
    except Exception as e:
        logging.error("join_game: Error joining game: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Internal server error") from e

@app.post("/create-game")
async def create_game(data: CreateGameData):
    """Handle a POST request to create a new game"""
    if not redis_client:
        logging.error("create_game: Redis is not available")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Internal database error")
    try:
        logging.debug("create-game data: %s", data)
        if data.max_players == '' or data.rounds == '':
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                                detail="Invalid rounds or max players")
        max_players = int(data.max_players)
        rounds = int(data.rounds)
        if (max_players < MIN_PLAYERS or max_players > MAX_PLAYERS
                or rounds < MIN_ROUNDS or rounds > MAX_ROUNDS):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                                detail="Invalid rounds or max players")

        if len(data.player_name) > MAX_LEN_PLAYER_NAME:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                                detail="Player name too long")

        game_id = str(uuid.uuid4())
        game_data = {
            "status": "waiting",
            "max_players": max_players,
            "rounds": rounds,
        }
        await redis_client.set(f"game:{game_id}", json.dumps(game_data))
        logging.info("Created new game with ID: %s", game_id)
        player_id = str(uuid.uuid4())
        player_data = {
            "name": data.player_name,
            "is_host": True,
        }
        await insert_player(game_id, player_id, player_data)
        return {"game_id": game_id,
                "player_id": player_id,
                "is_host": 'True'}
    except HTTPException:
        raise
    except Exception as e:
        logging.error("create_game: Error creating game: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Internal server error") from e

# These have to come after or the endpoints are overriden and incorrectly
# routed.
NEXT_DIR = "/quickdraw/out"
app.mount("/", StaticFiles(directory=NEXT_DIR, html = True), name="static")
