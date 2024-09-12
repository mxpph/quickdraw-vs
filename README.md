# quickdraw-vs

Quickdraw Versus is a web-based multiplayer guessing game, based on the single-player “Quick, Draw!” game developed by Google. All players are given a word to draw, and while they draw it, a machine learning model guesses, based on 16 categories, what they are drawing. The first player who’s drawing is correctly guessed by the model wins the round. The player who wins the most rounds wins the game.

The dataset used for this project is the Google Quick Draw dataset, made available by Google, Inc. under the Creative Commons Attribution 4.0 International license.
https://creativecommons.org/licenses/by/4.0/

## How to run

1. Install docker (https://www.docker.com/)
2. Run `docker-compose up --build`
3. The project should start at `localhost:3000`
