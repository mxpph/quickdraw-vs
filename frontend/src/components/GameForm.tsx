import { FormEvent, useState } from "react"
import Cookies from 'js-cookie'

export default function GameForm() {

  const [formData, setFormData] = useState({
    game_id: '',
    player_name: '',
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
      noValidate={true}
      onSubmit={handleSubmit}
      className="grid place-items-center shadow-md bg-neutral-50 rounded px-8 pt-6 pb-8 mb-4 gap-2"
    >
      <p>Game ID</p>
      <input
        type="text"
        className="rounded-md p-2 bg-neutral-200 shadow-md"
        name="game_id"
        placeholder="Blank for new game"
        onChange={handleChange}
        required
      />
      <p>Player Name</p>
      <input
        type="text"
        className="rounded-md p-2 bg-neutral-200 shadow-md"
        name="player_name"
        onChange={handleChange}
        required
      />
      <button
        type="submit"
        className="bg-blue-400 mt-4 font-semibold text-white rounded-lg p-2"
      >
        Create Game
      </button>
    </form>
  )
}
