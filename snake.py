import pygame
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

# Initialisation de Pygame
pygame.init()

# Définition des couleurs et dimensions
NOIR, BLANC, ROUGE, VERT = (0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0)
LARGEUR, HAUTEUR = 400, 400
TAILLE_BLOC = 20

# Initialisation de l'écran
ecran = pygame.display.set_mode((LARGEUR, HAUTEUR))
pygame.display.set_caption('Snake RL')

class SnakeEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.snake = [(LARGEUR//2, HAUTEUR//2)]
        self.direction = random.choice([(0, -TAILLE_BLOC), (0, TAILLE_BLOC), (-TAILLE_BLOC, 0), (TAILLE_BLOC, 0)])
        self.pomme = self.nouvelle_pomme()
        self.score = 0
        self.game_over = False
        return self.get_state()

    def nouvelle_pomme(self):
        while True:
            pomme = (random.randint(0, (LARGEUR-TAILLE_BLOC)//TAILLE_BLOC)*TAILLE_BLOC,
                     random.randint(0, (HAUTEUR-TAILLE_BLOC)//TAILLE_BLOC)*TAILLE_BLOC)
            if pomme not in self.snake:
                return pomme

    def get_state(self):
        head = self.snake[0]
        point_l = (head[0] - TAILLE_BLOC, head[1])
        point_r = (head[0] + TAILLE_BLOC, head[1])
        point_u = (head[0], head[1] - TAILLE_BLOC)
        point_d = (head[0], head[1] + TAILLE_BLOC)
        
        dir_l = self.direction == (-TAILLE_BLOC, 0)
        dir_r = self.direction == (TAILLE_BLOC, 0)
        dir_u = self.direction == (0, -TAILLE_BLOC)
        dir_d = self.direction == (0, TAILLE_BLOC)

        state = [
            (dir_l and self.collision(point_l)) or (dir_r and self.collision(point_r)) or (dir_u and self.collision(point_u)) or (dir_d and self.collision(point_d)),
            (dir_u and self.collision(point_l)) or (dir_d and self.collision(point_r)) or (dir_r and self.collision(point_u)) or (dir_l and self.collision(point_d)),
            (dir_d and self.collision(point_l)) or (dir_u and self.collision(point_r)) or (dir_l and self.collision(point_u)) or (dir_r and self.collision(point_d)),
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            self.pomme[0] < head[0],
            self.pomme[0] > head[0],
            self.pomme[1] < head[1],
            self.pomme[1] > head[1]
        ]
        return np.array(state, dtype=int)

    def collision(self, point):
        return point in self.snake or point[0] < 0 or point[0] >= LARGEUR or point[1] < 0 or point[1] >= HAUTEUR

    def step(self, action):
        if action == 0:  # Continuer tout droit
            pass
        elif action == 1:  # Tourner à gauche
            self.direction = (self.direction[1], -self.direction[0])
        elif action == 2:  # Tourner à droite
            self.direction = (-self.direction[1], self.direction[0])

        new_head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])
        
        if self.collision(new_head):
            self.game_over = True
            return self.get_state(), -10, True

        self.snake.insert(0, new_head)
        
        if new_head == self.pomme:
            self.score += 1
            self.pomme = self.nouvelle_pomme()
            reward = 10
        else:
            self.snake.pop()
            reward = 0

        return self.get_state(), reward, self.game_over

    def render(self):
        ecran.fill(NOIR)
        for segment in self.snake:
            pygame.draw.rect(ecran, VERT, [segment[0], segment[1], TAILLE_BLOC, TAILLE_BLOC])
        pygame.draw.rect(ecran, ROUGE, [self.pomme[0], self.pomme[1], TAILLE_BLOC, TAILLE_BLOC])
        pygame.display.update()

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((2**state_size, action_size))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def get_action(self, state):
        state_idx = self.state_to_index(state)
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        return np.argmax(self.q_table[state_idx])

    def state_to_index(self, state):
        return int(''.join(map(str, state)), 2)

    def train(self, state, action, reward, next_state, done):
        state_idx = self.state_to_index(state)
        next_state_idx = self.state_to_index(next_state)
        
        current_q = self.q_table[state_idx, action]
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state_idx])
        
        self.q_table[state_idx, action] += self.lr * (target_q - current_q)

        if not done:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

def train_agent(env, agent, episodes=10000, max_steps=1000):
    scores = []
    fig, (ax1, ax2, ax_slider) = plt.subplots(3, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [3, 1, 0.5]})
    line, = ax2.plot([], [])
    ax2.set_xlim(0, episodes)
    ax2.set_ylim(0, 100)
    ax2.set_xlabel('Épisodes')
    ax2.set_ylabel('Score')
    ax2.set_title('Performances de l\'agent')

    slider = Slider(ax_slider, 'Vitesse (épisodes/minute)', 1, 600, valinit=60, valstep=1)
    speed = 60  # Valeur initiale : 1 épisode par seconde

    def update_speed(val):
        nonlocal speed
        speed = val

    slider.on_changed(update_speed)

    def update_plot(frame):
        line.set_data(range(len(scores)), scores)
        ax2.relim()
        ax2.autoscale_view()
        return line,

    ani = FuncAnimation(fig, update_plot, interval=100, blit=True)
    plt.show(block=False)

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.train(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            if episode % 10 == 0:  # Render every 10 episodes to speed up training
                env.render()
            
            if done:
                break
        
        scores.append(total_reward)
        
        if episode % 100 == 0:
            print(f"Episode: {episode}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")
        
        plt.pause(60 / speed)  # Ajuster la pause en fonction de la vitesse

    return scores

# Initialisation
env = SnakeEnv()
state_size = len(env.get_state())
action_size = 3  # 0: continue straight, 1: turn left, 2: turn right
agent = QLearningAgent(state_size, action_size)

try:
    # Entraînement
    scores = train_agent(env, agent, episodes=10000)  # Augmenté à 10000 épisodes
except Exception as e:
    print(f"Une erreur s'est produite : {e}")
finally:
    # Fermeture de la fenêtre Pygame
    pygame.quit()

# Affichage du graphique final
plt.figure(figsize=(10, 5))
plt.plot(scores)
plt.title('Scores au fil des épisodes')
plt.xlabel('Épisodes')
plt.ylabel('Score')
plt.show()