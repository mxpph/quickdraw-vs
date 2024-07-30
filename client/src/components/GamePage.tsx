"use client";
import dynamic from "next/dynamic";

const Canvas = dynamic(() => import("./DrawCanvas"), {
  ssr: false,
});

export default function GamePage() {
  return (
    <div className="w-full grid place-items-center my-3">
      <div
        className="outline outline-2 outline-offset-2 outline-blue-500 rounded-xl overflow-hidden w-[90vw] shadow-xl grid place-items-center"
        id="canvasdiv"
      >
        <Canvas />
      </div>
    </div>
  );
}
