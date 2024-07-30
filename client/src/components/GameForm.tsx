let handleSubmit = (event: React.FormEvent<HTMLFormElement>): void => {
  event.preventDefault();

  const gameId = document.getElementById("game_id")?.textContent;
  const playerName = document.getElementById("player_name")?.textContent;

  const data = {
    game_id: gameId,
    player_name: playerName,
  };

  fetch("http://localhost:8000/create-game", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  })
    .then((response) => response.json())
    .then((result) => {
      console.log("Success:", result);
    })
    .catch((error) => {
      console.error("Error:", error);
    });
};

export default function GameForm() {
  {
    /* no validate = false in prod */
  }
  return (
    <form
      noValidate={true}
      onSubmit={handleSubmit}
      className="grid place-items-center shadow-md bg-neutral-50 rounded px-8 pt-6 pb-8 mb-4 gap-2"
      id="gameForm"
    >
      <p>Game ID</p>
      <input
        type="text"
        className="rounded-md p-2 bg-neutral-200 shadow-md"
        id="game_id"
        name="game_id"
        required
      />
      <p>Player Name</p>
      <input
        type="text"
        className="rounded-md p-2 bg-neutral-200 shadow-md"
        id="player_name"
        name="player_name"
        required
      />
      <button
        type="submit"
        className="bg-blue-400 mt-4 font-semibold text-white rounded-lg p-2"
      >
        Create Game
      </button>
    </form>
  );
}
