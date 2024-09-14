import GameForm from "@/components/GameForm";

export default function Home() {
  return (
    <main>
      <div className="w-full grid place-items-center my-3">
        <h1 id="test" className="text-3xl w-full text-center font-semibold py-2">
          Quickdraw Versus
        </h1>
        <GameForm />
      </div>
      <div className="mx-auto max-w-screen-sm px-5 justify-center space-y-4">
        <p className="text-justify">
          Quickdraw Versus is a multiplayer guessing game, based on the
          single-player “Quick, Draw!” game developed by Google, and using their
          dataset too.
        </p>
        <p className="text-justify">
          All players are given a word to draw, and while they draw it, a
          machine learning model guesses what they are drawing. The first player
          who’s drawing is correctly guessed by the model wins the round. The
          player who wins the most rounds wins the game. Have fun!
        </p>
      </div>
    </main>
  );
}
