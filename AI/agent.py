from game import GameController, MoveDirection
from collections import deque
import random
import torch
import numpy as np
from .neural_network import Linear_2048Qnet, QTrainer
from .utils import output2move, plot_training, print_game
from typing import Callable


class Agent2048:
    def __init__(
        self,
        game: GameController,
        get_state: Callable[[GameController], list],
        get_reward: Callable[[list, list, int, int, int, bool, bool], float],
        state_size: int,
        brain_structure=[],
        learning_rate=0.001,
        gamma=0.9,
        max_memory=100_000,
        batch_size=1000,
        optmizer=torch.optim.SGD,
        criterion=torch.nn.MSELoss,
        exploration_heavy_epochs=40,
        random_exploration_probability=0.5,
    ) -> None:
        self.game = game
        self.get_state = get_state
        self.get_reward = get_reward
        self.n_games = 0
        self.learning_rate = learning_rate
        self.exploration_heavy_epochs = exploration_heavy_epochs  # randomness
        self.random_exploration_proabibility = random_exploration_probability
        self.gamma = gamma  # discount rate
        self.batch_size = batch_size
        self.memory = deque(maxlen=max_memory)  # First-in-First-Out queue
        self.model = Linear_2048Qnet(
            input_size=state_size, hidden_layers=brain_structure
        )
        self.trainer = QTrainer(
            self.model,
            optimizer=optmizer,
            criterion=criterion,
            learning_rate=self.learning_rate,
            gamma=self.gamma,
        )

    def remember(
        self, state, action: MoveDirection, reward: float, next_state, game_over: bool
    ):
        self.memory.append((state, action, reward, next_state, game_over))

    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            sample = random.sample(self.memory, self.batch_size)
        else:
            sample = self.memory

        states, actions, rewards, next_states, game_overs = zip(*sample)
        return self.trainer.train_step(
            states, actions, rewards, next_states, game_overs
        )

    def train_short_memory(
        self, state, action: MoveDirection, reward: float, next_state, game_over: bool
    ):
        return self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state):
        # random moves : tradeoff exploration vs exploitation
        random_move_odds = (
            self.exploration_heavy_epochs - self.n_games
        ) * self.random_exploration_proabibility

        if random.random() < random_move_odds:
            move = self.game.get_random_move()
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model.forward(state0)
            move = output2move(prediction, self.game)

        return move

    def train(self, max_games=0, max_error=0.01):
        total_score = 0
        scores: list[int] = []
        mean_scores: list[int] = []

        total_error = 0
        errors: list[int] = []
        mean_errors: list[int] = []

        total_moves = 0
        moves: list[int] = []
        mean_moves: list[float] = []

        total_highest_block = 0
        highest_blocks: list[int] = []
        mean_highest_blocks: list[float] = []

        record = 0
        self.n_games = 0
        self.game.start()
        game_moves = 0
        error = np.Infinity
        print_game(self.game, f"Game {self.n_games + 1} (err: {error:.5f})")

        should_train = (
            lambda n_games, err: n_games < max_games and err > max_error
            if max_games > 0
            else lambda n_games, err: err > max_error
        )
        while should_train(self.n_games, error):
            # get current state
            state_current = self.get_state(self.game)

            # get move
            move = self.get_action(state_current)
            game_moves += 1

            # perform action to get new state
            points, blocks_moved, merges = self.game.move(move)

            state_new = self.get_state(self.game)

            print_game(self.game, f"Game {self.n_games + 1} (avg error: {error:.4f})")

            game_over = self.game.is_winner() or self.game.is_game_over()
            reward = self.get_reward(
                state_current,
                state_new,
                points,
                blocks_moved,
                merges,
                self.game.is_winner(),
                self.game.is_game_over(),
            )

            # train short memory
            self.train_short_memory(
                state_current, move.value, reward, state_new, game_over
            )

            # remember
            self.remember(state_current, move.value, reward, state_new, game_over)

            if game_over:
                # train long memory, plot result
                self.n_games += 1
                error = self.train_long_memory()
                score = self.game.score
                if score > record:
                    record = score
                    self.model.save()

                scores.append(score)
                total_score += score
                mean_score = total_score / self.n_games
                mean_scores.append(mean_score)

                errors.append(error)
                total_error += error
                mean_error = total_error / self.n_games
                mean_errors.append(mean_error)

                highest_block = self.game.board.highest_block()
                highest_blocks.append(highest_block)
                total_highest_block += highest_block
                mean_highest_block = float(total_highest_block) / self.n_games
                mean_highest_blocks.append(mean_highest_block)

                moves.append(game_moves)
                total_moves += game_moves
                mean_game_moves = float(total_moves) / self.n_games
                mean_moves.append(mean_game_moves)
                game_moves = 0

                self.game.start()

        plot_training(
            scores,
            mean_scores,
            errors,
            mean_errors,
            moves,
            mean_moves,
            highest_blocks,
            mean_highest_blocks,
        )
