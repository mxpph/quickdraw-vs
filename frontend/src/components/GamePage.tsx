"use client";
import { error } from "console";
import Link from "next/link"
import dynamic from "next/dynamic";
import Cookies from "js-cookie";
import { useState, useEffect, useRef } from "react";
import WaitingArea from "./WaitingArea";

const Canvas = dynamic(() => import("./DrawCanvas"), {
  ssr: false,
});

export default function GamePage() {


  const [errorShown, setErrorShown] = useState(false);
  const [hostButtonsShown, setHostButtonsShown] = useState(false);
  const [gameId, setGameId] = useState("")
  const [wordToGuess, setWordToGuess] = useState("");
  const [clearCanvas, setClearCanvas] = useState(false);
  const [gameWinner, setGameWinner] = useState("")

  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    // Check if the user is the host and set the state accordingly
    setHostButtonsShown(
      sessionStorage.getItem("quickdrawvs_is_host") === "True"
    );
    setGameWinner("")
    setGameId( Cookies.get("quickdrawvs_game_id") as string )

  }, []); // Adding an empty dependency array to ensure this runs only once on mount

  useEffect(() => {
    // Initialize WebSocket and store it in the ref
    ws.current = new WebSocket("ws://localhost:3000/ws");

    ws.current.onopen = (event) => {
      console.log("WebSocket connected!");
    };

    ws.current.onerror = (event) => {
      setErrorShown(true);
    };

    ws.current.onmessage = (event) => {
      console.log(event.data);
      try {
        const data = JSON.parse(event.data);
        switch (data.type) {
          case "next_round":
            setWordToGuess(data.word);
            handleClearCanvas()
            break;
          case "game_over":
            setGameWinner(data.winner)
            break;
        }
      } catch (error: any) {
        ws.current?.close();
        setErrorShown(true);
        ws.current = null;
      }
    };

    ws.current.onclose = (event) => {
      console.log("WebSocket closed with code", event.code);
    };

    // Clean up WebSocket when the component unmounts
    return () => {
      ws.current?.close();
    };
  }, []); // Empty dependency array to run once on mount

  const startGameMessage = () => {
    setHostButtonsShown(false);
    ws.current?.send('{"type": "start_game"}');
  };

  const winMessage = () => {
    ws.current?.send('{"type": "win"}');
  };

  const handlePredictionData = (data: string) => {
    if (data === wordToGuess) {
      winMessage();
    }
  };

  const handleClearCanvas = () => {
    setClearCanvas(true);
  };

  const onClearCanvas = () => {
    setClearCanvas(false);
  };


  return (
    <div className="mx-5">
      {errorShown && (
        <div role="alert" className="alert alert-error">
          <span>
            Error when connecting to the game server. Are you sure you have the
            right game ID?
          </span>
        </div>
      )}
      { !errorShown &&(
        <WaitingArea hostButtonsShown={hostButtonsShown} gameId={gameId} startGame={startGameMessage} />
      )}
      <button className="btn btn-primary absolute top-0 left-0 bg-opacity-50" onClick={winMessage}>
        Win round (dev button)
      </button>
      {(wordToGuess && gameWinner === "") && ( // !wordToGuess' only for development
        <div className="w-full grid place-items-center align-middle h-[95vh] mt-2">
          <h2 className="text-3xl mb-4 text-center justify-center align-middle h-[5vh]">Draw: <b className="font-semibold">{wordToGuess}</b></h2>
          <div
            className="outline outline-2 outline-offset-2 outline-primary rounded-xl overflow-hidden w-[90vw] h-full shadow-xl grid place-items-center"
            id="canvasdiv"
          >
            <Canvas dataPass={handlePredictionData} onParentClearCanvas={onClearCanvas} clearCanvas={clearCanvas} />
          </div>
        </div>
      )}
      {gameWinner !== "" && (
        <div className="w-full grid place-items-center">
          <div className="mt-10 rounded-2xl bg-neutral-50 grid place-items-center w-full max-w-3xl p-48 gap-2 align-middle shadow">
              <p className="text-xl">THE WINNER IS</p>
              <b className="text-xl font-bold mb-4">{gameWinner}!</b>
              <Link href="/" className="btn btn-primary" onClick={startGameMessage}>
                Return home
              </Link>
          </div>
        </div>
      )}
    </div>
  );
}
