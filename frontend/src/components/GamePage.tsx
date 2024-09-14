"use client";
import Link from "next/link";
import dynamic from "next/dynamic";
import Cookies from "js-cookie";
import { useState, useEffect, useRef } from "react";
import WaitingArea from "./WaitingArea";
import ErrorBar from "./ErrorBar";

const Canvas = dynamic(() => import("./DrawCanvas"), {
  ssr: false,
});

export default function GamePage() {
  const [errorMessage, setErrorMessage] = useState("");
  const [hostButtonsShown, setHostButtonsShown] = useState(false);
  const [gameId, setGameId] = useState("");
  const [wordToGuess, setWordToGuess] = useState("");
  const [clearCanvas, setClearCanvas] = useState(false);
  const [scoreboard, setScoreboard] = useState<string[]>([]);

  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    // Check if the user is the host and set the state accordingly
    setHostButtonsShown(
      sessionStorage.getItem("quickdrawvs_is_host") === "True"
    );
    setScoreboard([]);
    setGameId(Cookies.get("quickdrawvs_game_id") as string);
  }, []); // Adding an empty dependency array to ensure this runs only once on mount

  const handle_scoreboard = (scoreboardData: Array<[string, number]>) => {
    var tempScoreboard: string[] = [];
    for (var [player, score] of scoreboardData) {
      tempScoreboard.push(`${player}: ${score} point${score > 1 ? "s" : ""}`);
    }
    setScoreboard(tempScoreboard);
  };

  useEffect(() => {
    // Initialize WebSocket and store it in the ref
    ws.current = new WebSocket(`ws://${window.location.host}/ws/`);

    ws.current.onerror = (event) => {
      setErrorMessage(
        "Connection error. Are you sure you have the right game ID?"
      );
    };

    ws.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        switch (data.type) {
          case "next_round":
            setWordToGuess(data.word);
            handleClearCanvas();
            break;
          case "game_over":
            handle_scoreboard(data.scoreboard);
            break;
          case "cancel":
            setErrorMessage("The host disconnected, the game is cancelled");
            break;
        }
      } catch (error: any) {
        ws.current?.close();
        setErrorMessage("An unexpected error occurred");
        ws.current = null;
      }
    };

    ws.current.onclose = (event) => {
      console.log("Closed with code", event.code);
      if (event.code != 1000) {
        setErrorMessage(event.reason);
      }
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
      {errorMessage && (
        <ErrorBar error={errorMessage} setError={setErrorMessage} />
      )}
      {!errorMessage && !wordToGuess && (
        <WaitingArea
          hostButtonsShown={hostButtonsShown}
          gameId={gameId}
          startGame={startGameMessage}
        />
      )}
      {/* {<button
        className="btn btn-primary absolute top-0 left-0 bg-opacity-50"
        onClick={winMessage}
      >
        Win round (dev button)
      </button>} */}
      {wordToGuess && scoreboard.length === 0 && (
        <div className="w-full grid place-items-center align-middle h-[95vh] mt-2">
          <h2 className="text-3xl mb-4 text-center justify-center align-middle h-[5vh]">
            Draw: <b className="font-semibold">{wordToGuess}</b>
          </h2>
          <div
            className="outline outline-2 outline-offset-2 outline-primary rounded-xl overflow-hidden w-[90vw] h-full shadow-xl grid place-items-center"
            id="canvasdiv"
          >
            <Canvas
              dataPass={handlePredictionData}
              onParentClearCanvas={onClearCanvas}
              clearCanvas={clearCanvas}
            />
          </div>
        </div>
      )}
      {scoreboard.length > 0 && (
        <div className="w-full grid place-items-center">
          <div className="mt-10 rounded-2xl bg-neutral-50 grid place-items-center w-full max-w-3xl p-48 gap-2 align-middle shadow">
            <p className="text-xl justify-center">Final scores:</p>
            <ol className="list-decimal mb-4">
              <li className="text-lg animate-bounce">{scoreboard[0]}</li>
              {scoreboard.slice(1).map((str, i) => (
                <li key={i} className={"text-lg"}>
                  {str}
                </li>
              ))}
            </ol>
            <Link href="/" className="btn btn-primary mb-4">
              Return home
            </Link>
          </div>
        </div>
      )}
    </div>
  );
}
