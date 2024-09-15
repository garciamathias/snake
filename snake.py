import pygame
import numpy as np
import random

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

# Exemple d'utilisation
env = SnakeEnv()
clock = pygame.time.Clock()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    action = random.randint(0, 2)  # Action aléatoire pour cet exemple
    state, reward, done = env.step(action)
    env.render()
    clock.tick(10)

    if done:
        env.reset()