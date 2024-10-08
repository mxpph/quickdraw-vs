import { FormEvent, useState } from "react";
import Cookies from "js-cookie";
import ErrorBar from "./ErrorBar";

export default function GameForm() {
  const [createFormData, setCreateFormData] = useState({
    player_name: "",
    max_players: "",
    rounds: "",
  });

  const [joinFormData, setJoinFormData] = useState({
    game_id: "",
    player_name: "",
  });

  const [error, setError] = useState<string>("");

  const handleCreateFormChange = (e: any) => {
    setCreateFormData({
      ...createFormData,
      [e.target.name]: e.target.value,
    });
  };

  const handleJoinFormChange = (e: any) => {
    setJoinFormData({
      ...joinFormData,
      [e.target.name]: e.target.value,
    });
  };

  async function handleSubmit(event: FormEvent<HTMLFormElement>, form: string) {
    event.preventDefault();

    var url;
    var formData;
    if (form === "join") {
      url = "/api/join-game/";
      formData = joinFormData;
    } else {
      url = "/api/create-game/";
      formData = createFormData;
    }
    try {
      const response = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });
      if (response.ok) {
        const result = await response.json();
        const inFiveMinutes = new Date(new Date().getTime() + 5 * 60 * 1000);
        Cookies.set("quickdrawvs_game_id", result.game_id, {
          expires: inFiveMinutes,
        });
        Cookies.set("quickdrawvs_player_id", result.player_id, {
          expires: inFiveMinutes,
        });
        sessionStorage.setItem("quickdrawvs_is_host", result.is_host);
        window.location.href = `/game.html`;
      } else {
        const error = await response.json();
        throw new Error(error.detail);
      }
    } catch (error: any) {
      setError(error.message);
    }
  }

  return (
    <div className="grid gap-4">
      {error && <ErrorBar error={error} setError={setError} />}
      <div className="flex flex-row flex-wrap justify-center gap-4">
        <form
          id="createForm"
          onSubmit={(e: any) => handleSubmit(e, "create")}
          className="grid place-items-center shadow-md bg-neutral-50 rounded px-8 pt-6 pb-8 mb-4 gap-2"
        >
          <label className="form-control w-full max-w-xs">
            <div className="label">
              <span className="label-text">Player Name</span>
            </div>
            <input
              type="text"
              className="input input-bordered input-primary w-full max-w-xs"
              name="player_name"
              maxLength={16}
              onChange={handleCreateFormChange}
              required
            />
          </label>
          <div className="w-full grid grid-cols-2 grid-rows-1 gap-2">
            <select
              className="select select-primary"
              name="max_players"
              onChange={handleCreateFormChange}
              form="createForm"
            >
              <option disabled selected>
                Max players
              </option>
              <option>2</option>
              <option>3</option>
              <option>4</option>
              <option>5</option>
              <option>6</option>
            </select>
            <select
              className="select select-primary"
              name="rounds"
              onChange={handleCreateFormChange}
              form="createForm"
            >
              <option disabled selected>
                Rounds
              </option>
              <option>2</option>
              <option>3</option>
              <option>4</option>
              <option>5</option>
              <option>6</option>
              <option>7</option>
              <option>8</option>
              <option>9</option>
              <option>10</option>
            </select>
          </div>
          <button type="submit" className="btn btn-primary">
            Create Game
          </button>
        </form>
        <form
          id="joinForm"
          onSubmit={(e: any) => handleSubmit(e, "join")}
          className="grid place-items-center shadow-md bg-neutral-50 rounded px-8 pt-6 pb-8 mb-4 gap-2"
        >
          <label className="form-control w-full max-w-xs">
            <div className="label">
              <span className="label-text">Game ID</span>
            </div>
            <input
              type="text"
              className="input input-bordered input-primary w-full max-w-xs"
              name="game_id"
              maxLength={5}
              onChange={handleJoinFormChange}
              required
            />
          </label>
          <label className="form-control w-full max-w-xs">
            <div className="label">
              <span className="label-text">Player Name</span>
            </div>
            <input
              type="text"
              className="input input-bordered input-primary w-full max-w-xs"
              name="player_name"
              maxLength={16}
              onChange={handleJoinFormChange}
              required
            />
          </label>
          <button type="submit" className="btn btn-primary">
            Join Game
          </button>
        </form>
      </div>
    </div>
  );
}
