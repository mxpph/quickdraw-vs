import { FormEvent, useState } from "react"
import Cookies from 'js-cookie'

export default function GameForm() {

  const [formData, setFormData] = useState({
    game_id: '',
    player_name: '',
    max_players: '',
    rounds: '',
  })

  const handleChange = (e: any) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    })
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()

    try {
      const response = await fetch("http://localhost:3000/create-game", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      })
      if (response.ok) {
        const result = await response.json()
        const inFiveMinutes = new Date(new Date().getTime() + 5 * 60 * 1000)
        Cookies.set('quickdrawvs_game_id', result.game_id, {
            expires: inFiveMinutes
        })
        Cookies.set('quickdrawvs_player_id', result.player_id, {
            expires: inFiveMinutes
        })
        sessionStorage.setItem("quickdrawvs_is_host", result.is_host)
        window.location.href = `/game.html`
      } else {
        const errorText = await response.text()
        throw new Error(`Network response was not ok: ${response.statusText}, ${errorText}`)
      }
    } catch (error) {
      console.error("Error:", error)
    }
  }
  return (
    <form
      id="gameform"
      noValidate={true}
      onSubmit={handleSubmit}
      className="grid place-items-center shadow-md bg-neutral-50 rounded px-8 pt-6 pb-8 mb-4 gap-2"
    >
      <label className="form-control w-full max-w-xs">
        <div className="label">
          <span className="label-text">Game ID</span>
          <span className="label-text-alt">(blank for new game)</span>
        </div>
        <input
          type="text"
          className="input input-bordered input-primary w-full max-w-xs"
          name="game_id"
          maxLength={36}
          onChange={handleChange}
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
          onChange={handleChange}
          required
        />
      </label>
      <div className="w-full grid grid-cols-2 grid-rows-1 gap-2">
        <select
          className="select select-primary"
          name="max_players"
          onChange={handleChange}
          form="gameform"
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
          onChange={handleChange}
          form="gameform"
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
      <button
        type="submit"
        className="btn btn-primary"
      >
        {/* TODO: Separate join/create forms */}
        Join Game
      </button>
    </form>
  );
}
