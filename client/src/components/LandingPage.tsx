'use client';
import dynamic from 'next/dynamic';

const Canvas = dynamic(() => import('./DrawCanvas'), {
  ssr: false,
});

export default function LandingPage() {
    return(
        <div className="w-full">
            <h1 id="test" className="text-3xl w-full text-center font-semibold py-2">quickdraw vs</h1>
            <div className="w-full grid place-items-center mb-6">
                <button className="place-items-center bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded mb-4" id="newgame">
                    New game
                </button>
                <div className="outline outline-2 outline-offset-2 outline-blue-500 rounded-xl overflow-hidden w-[90vw] shadow-xl grid place-items-center" id="canvasdiv">
                    <Canvas />
                </div>
            </div>
        </div>
    )
}