from copy import copy
import functools
import random
import cv2
import pygame
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box
from models.animal import Animal
from models.plant import Plant

WINDOW_WIDTH = 300
WINDOW_HEIGHT = 300
WHITE = (255,255,255)
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
BLACK = (0,0,0)
PX = WINDOW_WIDTH//2
PY = WINDOW_HEIGHT//2
SPEED_FACTOR = 1
DT_FACTOR = 0.05
PADDING_WIDTH = 40
LAG_THRESHOLD = 0.1 + DT_FACTOR
SCREEN_SHOT_WIDTH = 100
SCREEN_SHOT_HEIGHT = 100
BOUNDRY_WIDTH = 10
RENDER_MODE = True
NUM_PLANTS = 25
OBSERVATION_SCREENSHOT_WIDTH = 32
OBSERVATION_SCREENSHOT_HEIGHT = 32
FEATURES_SIZE = 3

class AnimalEnv(gym.Env):
    metadata = {
        "name": "AnimalEnv",
    }

    def __init__(self, window_width=WINDOW_WIDTH, window_height=WINDOW_HEIGHT, num_plants=NUM_PLANTS, speed_factor=SPEED_FACTOR, dt_factor=DT_FACTOR):
        self.window_width = window_width
        self.window_height = window_height
        self.num_plants = num_plants
        self.speed_factor = speed_factor
        self.dt_factor = dt_factor
        self.render_mode = RENDER_MODE
        self.steps = 0
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBSERVATION_SCREENSHOT_HEIGHT*OBSERVATION_SCREENSHOT_WIDTH*3+FEATURES_SIZE,),
            dtype=np.float32
        )
        self.action_space = Discrete(5)
        self.agent =  Animal(random.randint(BOUNDRY_WIDTH+PADDING_WIDTH, self.window_width-BOUNDRY_WIDTH), random.randint(BOUNDRY_WIDTH+PADDING_WIDTH, self.window_height-BOUNDRY_WIDTH), self.window_width, self.window_height, BLUE)

        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width+2*PADDING_WIDTH, self.window_height+2*PADDING_WIDTH))
        pygame.display.set_caption("Single Agent Environment")
        self.clock = pygame.time.Clock()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.screen.fill(BLACK)
        self.steps = 0
        self.plants = [Plant(random.randint(BOUNDRY_WIDTH+PADDING_WIDTH, self.window_width-BOUNDRY_WIDTH), random.randint(BOUNDRY_WIDTH+PADDING_WIDTH, self.window_height-BOUNDRY_WIDTH), (0, 255, 0), (255, 0, 0)) for _ in range(self.num_plants)]
        self.agent = Animal(random.randint(BOUNDRY_WIDTH+PADDING_WIDTH, self.window_width-BOUNDRY_WIDTH), random.randint(BOUNDRY_WIDTH+PADDING_WIDTH, self.window_height-BOUNDRY_WIDTH), self.window_width, self.window_height, BLUE)
        self.render
        self.timestep = 0

        observations = self.get_obs()
        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {}

        return observations, infos

    def step(self, action):
        agent = self.agent
        self.render()
        self.steps += 1
        dt = self.clock.tick(60 * self.speed_factor) / 1000 * self.speed_factor + self.dt_factor        
        terminations = False
        reward = 0
        truncation = False

        if dt < LAG_THRESHOLD + DT_FACTOR:
            agent.update(action, dt)
            
            shortest_distance = None
            for plant in self.plants:
                plant.grow()
                # find the nearest plant distance to the agent
                # distance = pygame.math.Vector2.distance_to(agent.position, plant.position)
                # if shortest_distance is None:
                #     shortest_distance = distance
                # elif distance < shortest_distance:
                #     shortest_distance = distance
                # #give the agent a reward for getting closer to the plant
                # if distance < 100:
                #     rewards[agent_name] += 100 - distance

                if agent.hit(plant):
                    reward += 10
                    if agent.eat(plant):
                        reward += 700
                    break
            if not agent.alive:
                print('died')
                reward -= 2000  # Large penalty for dying
            else:
                reward += 1  # Small reward for staying alive
            # if agent.position.x < BOUNDRY_WIDTH:
            #     rewards[agent_name] -= 2  # Penalty for hitting the wall
            #     if action == 3:
            #         rewards[agent_name] += 2
            # if agent.position.x > WINDOW_WIDTH - BOUNDRY_WIDTH:
            #     rewards[agent_name] -= 2
            #     if action == 2:
            #         rewards[agent_name] += 2
            # if agent.position.y < BOUNDRY_WIDTH:
            #     rewards[agent_name] -= 2
            #     if action == 1:
            #         rewards[agent_name] += 2
            # if agent.position.y > WINDOW_HEIGHT - BOUNDRY_WIDTH:
            #     rewards[agent_name] -= 2
            #     if action == 0:
            #         rewards[agent_name] += 2
            energy = agent.energy
            max_energy = agent.max_energy
            reward += energy / max_energy
            # get agent speed, punish for moving too fast
            # speed = math.sqrt(agent.x_speed ** 2 + agent.y_speed ** 2)
            # rewards[agent_name] -=  (speed / agent.max_speed * 50) / 10
        else:
            print('skipping frame')

        observations = self.get_obs()
        self.render_human_only()
     
        if not self.agent.alive:
            truncation = True
            
        info = {}  # Additional info dictionary


        return observations, reward, terminations, truncation, info


    def render(self):

        if RENDER_MODE:
            self.screen.fill(BLACK)
            inner_rect = pygame.Rect(
                PADDING_WIDTH,
                PADDING_WIDTH,
                self.window_width,
                self.window_height
            )
            pygame.draw.rect(self.screen, WHITE, inner_rect)  # Fill the middle with white

            self.handle_events()
            # self.screen.fill(WHITE)
            self.agent.draw(self.screen)
            for plant in self.plants:
                plant.draw(self.screen)
            #pygame.display.update()

    def render_human_only(self):
        font = pygame.font.Font(None, 20)
        if RENDER_MODE:
            fps = self.clock.get_fps()
            if self.steps % 100 == 0:
                print(f"FPS: {fps}")

            self.agent.draw_name(self.screen, font, "Agent")
            self.agent.draw_energy(self.screen)
            pygame.display.update()


    def get_obs(self):
        agent = self.agent
        image = agent.get_screenshot(self.screen, SCREEN_SHOT_WIDTH, SCREEN_SHOT_HEIGHT)
        image = self.show_snapshot(image)
        image = self.preprocess_image(image)
        features = self.get_features()
        obs = np.concatenate([image.flatten(), features])
        observations = np.array(obs, dtype=np.float32).flatten()
        return observations

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

    def get_features(self):
        agent = self.agent
        return np.array([agent.x_speed/agent.max_speed, agent.y_speed/agent.max_speed, agent.energy/agent.max_energy], dtype=np.float32)
                
    def show_snapshot(self, image):
        target_size = (32, 32)
        image = 255 - image
        image = cv2.resize(image, target_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # window_name = f"{agent_name} snapshot"
        # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # cv2.resizeWindow(window_name, 256, 256)
        # if self.steps % 5 == 0:
        #     cv2.imshow(f"{agent_name} snapshot", image)
        #     cv2.waitKey(1)  # Add a small delay to allow the image to be displayed
        
        return image
    
    def preprocess_image(self, image):
        image = image.astype(np.float32) / 255.0  # Normalize the image
        # image = np.transpose(image, (2, 0, 1))  # Transpose to (3, 64, 64)

        return image