'use client';
import dynamic from 'next/dynamic';

const Canvas = dynamic(() => import('./DrawCanvas'), {
  ssr: false,
});

export default function LandingPage() {
    return(
        <div className="w-full">
            <h1 id="test" className="text-3xl w-full text-center font-semibold py-2">quickdraw vs</h1>
            <form className="grid place-items-center shadow-md rounded px-8 pt-6 pb-8 mb-4 gap-2" id="gameForm">
                <p>Game ID</p>
                <input type="text" className='rounded-md p-2 bg-neutral-100 border-gray-500' id="game_id" name="game_id" required />
                <p>Player Name</p>
                <input type="text" className='rounded-md p-2 bg-neutral-100 border-gray-500' id="player_name" name="player_name" required />
                <button type="submit" className="bg-blue-400 rounded-lg p-2">Create Game</button>
            </form>
            <div className="w-full grid place-items-center mb-6">
                <div className="outline outline-2 outline-offset-2 outline-blue-500 rounded-xl overflow-hidden w-[90vw] shadow-xl grid place-items-center" id="canvasdiv">
                    <Canvas />
                </div>
            </div>
        </div>
    )
}