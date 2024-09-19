import pygame
import numpy as np
from snake_game import SnakeGame, Direction, Point
from agent import Agent

def get_state(game):
    head = game.snake[0]
    point_l = Point(head.x - 20, head.y)
    point_r = Point(head.x + 20, head.y)
    point_u = Point(head.x, head.y - 20)
    point_d = Point(head.x, head.y + 20)
    
    dir_l = game.direction == Direction.LEFT
    dir_r = game.direction == Direction.RIGHT
    dir_u = game.direction == Direction.UP
    dir_d = game.direction == Direction.DOWN

    state = [
        # Danger straight
        (dir_r and game.is_collision(point_r)) or 
        (dir_l and game.is_collision(point_l)) or 
        (dir_u and game.is_collision(point_u)) or 
        (dir_d and game.is_collision(point_d)),

        # Danger right
        (dir_u and game.is_collision(point_r)) or 
        (dir_d and game.is_collision(point_l)) or 
        (dir_l and game.is_collision(point_u)) or 
        (dir_r and game.is_collision(point_d)),

        # Danger left
        (dir_d and game.is_collision(point_r)) or 
        (dir_u and game.is_collision(point_l)) or 
        (dir_r and game.is_collision(point_u)) or 
        (dir_l and game.is_collision(point_d)),
        
        # Move direction
        dir_l,
        dir_r,
        dir_u,
        dir_d,
        
        # Food location 
        game.food.x < game.head.x,  # food left
        game.food.x > game.head.x,  # food right
        game.food.y < game.head.y,  # food up
        game.food.y > game.head.y  # food down
        ]

    return np.array(state, dtype=int)

def train():
    n_games = 1000
    batch_size = 32
    agent = Agent(state_size=11, action_size=4)
    game = SnakeGame()
    scores = []
    record = 0

    for i in range(n_games):
        game.reset()
        state = get_state(game)
        done = False
        score = 0
        moves_without_progress = 0

        while not done:
            action = agent.act(state)
            
            # Map action (0,1,2,3) to direction
            if action == 0:
                game.direction = Direction.LEFT
            elif action == 1:
                game.direction = Direction.RIGHT
            elif action == 2:
                game.direction = Direction.UP
            else:
                game.direction = Direction.DOWN
            
            prev_score = game.score
            done, score = game.play_step()
            next_state = get_state(game)
            
            if score > prev_score:
                moves_without_progress = 0
            else:
                moves_without_progress += 1

            if moves_without_progress >= 30:
                done = True
                reward = -10
            else:
                reward = 10 if score > prev_score else -10 if done else 0
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            agent.replay(batch_size)

        scores.append(score)
        mean_score = np.mean(scores[-100:])
        
        if score > record:
            record = score
            agent.save('best_agent.pth')
        
        print(f'Game {i+1}, Score: {score}, Record: {record}, Mean Score: {mean_score:.2f}')

    agent.save('final_agent.pth')

if __name__ == "__main__":
    train()