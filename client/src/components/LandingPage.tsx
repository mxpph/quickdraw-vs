'use client';
import dynamic from 'next/dynamic';

import GameForm from "./GameForm"

const Canvas = dynamic(() => import('./DrawCanvas'), {
  ssr: false,
});

export default function LandingPage() {
    return(
        <div className="w-full grid place-items-center">
            <h1 id="test" className="text-3xl w-full text-center font-semibold py-2">quickdraw vs</h1>
            <GameForm />
            <div className="w-full grid place-items-center mb-6">
                <div className="outline outline-2 outline-offset-2 outline-blue-500 rounded-xl overflow-hidden w-[90vw] shadow-xl grid place-items-center" id="canvasdiv">
                    <Canvas />
                </div>
            </div>
        </div>
    )
}