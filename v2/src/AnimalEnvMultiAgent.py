from copy import copy
import functools
import random
import cv2
import pygame
import numpy as np
from gymnasium.spaces import Discrete


#-----PLEASE EDIT CONSTANTS IN THE TRAINING SCRIPT, BELOW ARE DEFAULT VALUES-----#


# The size of the env window
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 600

WHITE = (255,255,255)
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)
BLACK = (0,0,0)


# The middle of the window
PX = WINDOW_WIDTH//2
PY = WINDOW_HEIGHT//2

# How fast the game runs
SPEED_FACTOR = 1
DT_FACTOR = 10

# Size of the padding of the window
PADDING_WIDTH = 40

# The lag threshold. If the lag is greater than this, skip the frame
LAG_THRESHOLD = 0.1 + DT_FACTOR

# Size of the screenshot the agent sees
SCREEN_SHOT_WIDTH = 100
SCREEN_SHOT_HEIGHT = 100

# The resolution of the screenshot
SCREEN_SHOT_RESOLUTION = (16, 16)

# The width of the boundary for the agent to spawn within
BOUNDRY_WIDTH = 10

RENDER_MODE = "Single"
NUM_PLANTS = 25

# The class that defines the environment the agents are in
class AnimalEnv():
    
    def __init__(self, agent_size, window_width=WINDOW_WIDTH, window_height=WINDOW_HEIGHT, image_shape=SCREEN_SHOT_RESOLUTION, num_plants=NUM_PLANTS, speed_factor=SPEED_FACTOR, dt_factor=DT_FACTOR, render_mode=RENDER_MODE):
        self.agent_size = agent_size
        self.window_width = window_width
        self.window_height = window_height
        self.num_plants = num_plants
        self.speed_factor = speed_factor
        self.dt_factor = dt_factor
        self.render_mode = render_mode
        self.image_shape = image_shape
        if len(image_shape) != 2:
            self.image_shape = image_shape[:2]
        self.steps = 0

        pygame.init()
        self.screen = pygame.display.set_mode((self.window_width+2*PADDING_WIDTH, self.window_height+2*PADDING_WIDTH))
        pygame.display.set_caption("Multi-Agent Environment")
        self.clock = pygame.time.Clock()

    
    def reset(self, agents=None, Plants=None, possible_agents=None):
        self.agents = []
        self.screen.fill(BLACK)
        self.steps = 0
        # create the agent instances
        self.agent_instances = agents
        self.plants = Plants
        for plant in self.plants:
            plant.land(random.randint(BOUNDRY_WIDTH+PADDING_WIDTH, self.window_width-BOUNDRY_WIDTH), random.randint(BOUNDRY_WIDTH+PADDING_WIDTH, self.window_height-BOUNDRY_WIDTH), )
        for agent_name in self.agent_instances:
            agent = self.agent_instances[agent_name]
            agent.land(random.randint(BOUNDRY_WIDTH+PADDING_WIDTH, self.window_width-BOUNDRY_WIDTH), random.randint(BOUNDRY_WIDTH+PADDING_WIDTH, self.window_height-BOUNDRY_WIDTH))
            agent.energy = 100
            agent.alive = True
        self.render
        self.agents = copy(possible_agents)
        self.timestep = 0
        observations = {
            a: (
                self.get_obs(self.agent_instances[a], a)            
                )
            for a in self.agents
        }
        return observations

    # The step function that runs each frame
    def step(self, actions):
        skip = False
        self.render()
        self.steps += 1
        dt = self.clock.tick(60 * self.speed_factor) / 1000 * self.speed_factor + self.dt_factor        
        terminations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}
        truncations = {a: False for a in self.agents}

        if dt < LAG_THRESHOLD + DT_FACTOR:
            # What happens each frame and reward calculation
            for plant in self.plants:
                plant.grow()

            for agent_name in self.agents:
                agent = self.agent_instances[agent_name]
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
        # If the lag is too high, skip the frame
        else:
            skip = True
            print('skipping frame')

        observations = {
            a: (
                self.get_obs(self.agent_instances[a], a)            
                )
            for a in self.agents
        }
        
        self.render_human_only()
        agent_names = {
            a: a
            for a in self.agents
        }
     
        for agent_name in self.agents:
            if not self.agent_instances[agent_name].alive:
                self.agents.remove(agent_name)

        return observations, rewards, terminations, truncations, skip, agent_names

    # The render function that draws the agents and plants
    def render(self):

        if self.render_mode == "Human" or self.render_mode == "Single":
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
                self.agent_instances[agent].draw(self.screen)
            for plant in self.plants:
                plant.draw(self.screen)
            #pygame.display.update()

    def render_human_only(self):
        font = pygame.font.Font(None, 20)
        if self.render_mode == "Human":
            for agent in self.agents:
                agent_ins = self.agent_instance_mapping[agent]
                agent_ins.draw_name(self.screen, font, agent)
                agent_ins.draw_energy(self.screen)
        pygame.display.update()

    
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
        target_size = self.image_shape
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


        return image
