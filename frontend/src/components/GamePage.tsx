"use client";
import dynamic from "next/dynamic";
import { useState } from "react";

const Canvas = dynamic(() => import("./DrawCanvas"), {
  ssr: false,
});

export default function GamePage() {
  const [errorShown, setErrorShown] = useState(false)

  const socket = new WebSocket("ws://localhost:3000/ws")

  socket.onopen = (event) => {
    console.log("WebSocket connected!")
  }

  socket.onerror = (event) => {
    setErrorShown(true)
  }

  socket.onclose = (event) => {
    console.log("WebSocket closed with code", event.code)
  }

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
