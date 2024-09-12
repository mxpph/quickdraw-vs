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
  return (
    <div className="w-full grid place-items-center h-full">
      <div className="mt-10 rounded-2xl bg-neutral-50 grid place-items-center w-full max-w-6xl p-28 gap-2 align-middle shadow">
        <p className="text-xl">Game ID</p>
        <p className="text-xl font-semibold mb-2">{gameId}</p>
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
