"use client";
import dynamic from "next/dynamic";
import { useState, useEffect, useRef } from "react";

const Canvas = dynamic(() => import("./DrawCanvas"), {
  ssr: false,
});

export default function GamePage() {
  const [errorShown, setErrorShown] = useState(false);
  const [hostButtonsShown, setHostButtonsShown] = useState(false);

  const ws = useRef<WebSocket | null>(null)

  useEffect(() => {
    // Check if the user is the host and set the state accordingly
    setHostButtonsShown(
      sessionStorage.getItem("quickdrawvs_is_host") === "True"
    );
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
      {!errorShown && hostButtonsShown && (
        <button
          className="btn btn-primary"
          onClick={startGameMessage}
        >
          Start game
        </button>
      )}
      <button
        className="btn btn-primary"
        onClick={winMessage}
      >
        Win round (dev button)
      </button>
      <div className="w-full grid place-items-center my-3">
        <div
          className="outline outline-2 outline-offset-2 outline-primary rounded-xl overflow-hidden w-[90vw] shadow-xl grid place-items-center"
          id="canvasdiv"
        >
          <Canvas />
        </div>
      </div>
    </div>
  );
}
