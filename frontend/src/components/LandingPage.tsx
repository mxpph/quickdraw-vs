import GameForm from "./GameForm";

export default function LandingPage() {
  return (
    <div className="w-full grid place-items-center my-3">
      <h1 id="test" className="text-3xl w-full text-center font-semibold py-2">
        quickdraw vs
      </h1>
      <GameForm />
    </div>
  );
}
