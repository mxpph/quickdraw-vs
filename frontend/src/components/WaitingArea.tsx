interface WaitingAreaProps {
  gameId: string;
  startGame: () => void;
  hostButtonsShown: boolean;
}

export default function WaitingArea({
  gameId,
  startGame,
  hostButtonsShown,
}: WaitingAreaProps) {

    async function handleCopy() {
        await navigator.clipboard.writeText(gameId);
    }

  return (
    <div className="w-full grid place-items-center h-full">
      <div className="mt-10 rounded-2xl bg-neutral-50 grid place-items-center w-full max-w-6xl p-28 align-middle shadow">
        <p className="text-xl mb-2">Game ID</p>
        <button className="mb-8" onClick={handleCopy}>
            <p className="text-xl font-semibold">{gameId}</p>
            <p className="text-neutral-400">Click to copy</p>
        </button>
        {hostButtonsShown ? (
          <button className="btn btn-primary" onClick={startGame}>
            Start game
          </button>
        ) : (
          <p>Waiting for host to start...</p>
        )}
      </div>
    </div>
  );
}
