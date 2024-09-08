"use client";
import { error } from "console";
import dynamic from "next/dynamic";
import Cookies from "js-cookie";
import { useState, useEffect, useRef } from "react";

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
            // TODO: handle this
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
    <div className="my-3 mx-5">
      {errorShown && (
        <div role="alert" className="alert alert-error">
          <span>
            Error when connecting to the game server. Are you sure you have the
            right game ID?
          </span>
        </div>
      )}
      { !errorShown && hostButtonsShown  &&(
        <div className="w-full grid place-items-center">
          <div className="mt-10 rounded-2xl bg-neutral-50 grid place-items-center w-full max-w-lg p-48 gap-4 align-middle shadow">
              <p className="text-xl">Game ID: {gameId}</p>
              <button className="btn btn-primary" onClick={startGameMessage}>
                Start game
              </button>
          </div>
        </div>
      )}
      <button className="btn btn-primary" onClick={winMessage}>
        Win round (dev button)
      </button>
      {(wordToGuess) && ( // '|| !wordToGuess' only for development
        <div className="w-full grid place-items-center my-3">
          <h2 className="text-3xl">Draw: {wordToGuess}</h2>
          <div
            className="outline outline-2 outline-offset-2 outline-primary rounded-xl overflow-hidden w-[90vw] shadow-xl grid place-items-center"
            id="canvasdiv"
          >
            <Canvas dataPass={handlePredictionData} onParentClearCanvas={onClearCanvas} clearCanvas={clearCanvas} />
          </div>
          <button className="my-2 mx-1 rounded-xl shadow shadow-neutral-400 px-2 bg-neutral-100 py-1" onClick={handleClearCanvas}>
            Clear Canvas From GamePage.tsx
          </button>
        </div>
      )}
    </div>
  );
}
