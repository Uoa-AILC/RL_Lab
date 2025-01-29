from copy import copy
import functools
import random
import cv2
import pygame
import numpy as np
from gymnasium.spaces import Discrete, Dict, Box
from ..models.animal import Animal
from ..models.plant import Plant

WINDOW_WIDTH = 600
WINDOW_HEIGHT = 600
WHITE = (255,255,255)
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
BLACK = (0,0,0)
PX = WINDOW_WIDTH//2
PY = WINDOW_HEIGHT//2
SPEED_FACTOR = 3
DT_FACTOR = 10
PADDING_WIDTH = 40
LAG_THRESHOLD = 0.1 + DT_FACTOR
SCREEN_SHOT_WIDTH = 100
SCREEN_SHOT_HEIGHT = 100
BOUNDRY_WIDTH = 10
RENDER_MODE = True
NUM_PLANTS = 25


class AnimalEnv():
    metadata = {
        "name": "AnimalEnv",
    }

    def __init__(self, agent_size, window_width=WINDOW_WIDTH, window_height=WINDOW_HEIGHT, num_plants=NUM_PLANTS, speed_factor=SPEED_FACTOR, dt_factor=DT_FACTOR):
        self.agent_size = agent_size
        self.window_width = window_width
        self.window_height = window_height
        self.num_plants = num_plants
        self.speed_factor = speed_factor
        self.dt_factor = dt_factor
        self.possible_agents = ["agent_" + str(r) for r in range(self.agent_size)]
        self.render_mode = RENDER_MODE
        self.steps = 0
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width+2*PADDING_WIDTH, self.window_height+2*PADDING_WIDTH))
        pygame.display.set_caption("Multi-Agent Environment")
        self.clock = pygame.time.Clock()


    def reset(self, seed=None, options=None):
        self.screen.fill(BLACK)
        self.steps = 0
        self.agents = []
        self.agent_instances = [
            Animal(random.randint(BOUNDRY_WIDTH+PADDING_WIDTH, self.window_width-BOUNDRY_WIDTH),
                   random.randint(BOUNDRY_WIDTH+PADDING_WIDTH, self.window_height-BOUNDRY_WIDTH),
                   self.window_width, self.window_height, BLUE)
            for _ in range(self.agent_size)
        ]

        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        
        self.agent_instance_mapping = dict(
            zip(self.possible_agents, self.agent_instances)
        )
        self.plants = [Plant(random.randint(BOUNDRY_WIDTH+PADDING_WIDTH, self.window_width-BOUNDRY_WIDTH), random.randint(BOUNDRY_WIDTH+PADDING_WIDTH, self.window_height-BOUNDRY_WIDTH), (0, 255, 0), (255, 0, 0)) for _ in range(self.num_plants)]

        self.render
        self.agents = copy(self.possible_agents)
        self.timestep = 0

        observations = {
            a: (
                self.get_obs(self.agent_instance_mapping[a], a)            
                )
            for a in self.agents
        }
        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {a: {} for a in self.agents}

        return observations, infos

    def step(self, actions):
        skip = False
        self.render()
        self.steps += 1
        dt = self.clock.tick(60 * self.speed_factor) / 1000 * self.speed_factor + self.dt_factor        
        terminations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}
        truncations = {a: False for a in self.agents}

        if dt < LAG_THRESHOLD + DT_FACTOR:
            for plant in self.plants:
                plant.grow()

            for agent_name in self.agents:
                agent = self.agent_instance_mapping[agent_name]
                action = actions[agent_name]
                agent.update(action, dt)
                rewards[agent_name] = 0
                
                for plant in self.plants:
                    if agent.eat(plant):
                        rewards[agent_name] += 2000
                        break
                if not agent.alive:
                    # print(f'{agent_name} died')
                    rewards[agent_name] -= 5000  # Large penalty for dying
                    
                max_energy = agent.max_energy
                rewards[agent_name] += agent.energy / max_energy

        else:
            skip = True
            print('skipping frame')

        observations = {
            a: (
                self.get_obs(self.agent_instance_mapping[a], a)            
                )
            for a in self.agents
        }
        
        self.render_human_only()
        agent_names = {
            a: a
            for a in self.agents
        }
     
        for agent_name in self.agents:
            if not self.agent_instance_mapping[agent_name].alive:
                truncations[agent_name] = True
                self.agents.remove(agent_name)
        if any(terminations.values()) or all(truncations.values()):
            self.agents = []

        return observations, rewards, terminations, truncations, skip, agent_names


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
            for agent in self.agent_instances:
                agent.draw(self.screen)
            for plant in self.plants:
                plant.draw(self.screen)
            #pygame.display.update()

    def render_human_only(self):
        font = pygame.font.Font(None, 20)
        if RENDER_MODE:
            for agent in self.agents:
                agent_ins = self.agent_instance_mapping[agent]
                agent_ins.draw_name(self.screen, font, agent)
                agent_ins.draw_energy(self.screen)
            pygame.display.update()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        self._observation_space = Dict({
            f"agent_{i}": Dict({
                'image': Box(low=0, high=1, shape=(3, 64, 64), dtype=np.float32),
                'features': Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
            }) for i in range(self.agent_size)
        })
        return self._observation_space[agent]
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(5)




    def get_obs(self, agent, agent_name):
        image = agent.get_screenshot(self.screen, SCREEN_SHOT_WIDTH, SCREEN_SHOT_HEIGHT)
        image = self.show_snapshot(image, agent_name)
        image = self.preprocess_image(image)
        features = self.get_features(agent)
        return np.concatenate([image.flatten(), features])

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

    def get_features(self, agent):
        return np.array([agent.x_speed/agent.max_speed, agent.y_speed/agent.max_speed, agent.energy/agent.max_energy], dtype=np.float32)
                
    def show_snapshot(self, image, agent_name):
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
