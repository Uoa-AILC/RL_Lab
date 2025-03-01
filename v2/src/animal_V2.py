import math
import random
import pygame
import numpy as np
import torch


BLUE = (0, 0, 255)
PINK = (255, 105, 180)
# The maximum speed the animal can move
MAX_SPEED = 100
# The maximum energy the animal can have
MAX_ENERGY = 100
# The maximum distance the animal can move in one frame
MAX_DISTANCE_PER_FRAME = 1000
# The acceleration of the animal
ACCELERATION = 100
# The start energy of the animal
START_ENERGY = 100

DRAG_COEFFICIENT = 0.98

MOVEMENT_ENERGY_COST = 0.4

LAG_THRESHOLD = 0.2

ENERGY_COST_PER_FRAME = 0.4

BASE_RADIUS = 5

class Animal:
    def __init__(self, x, y, width, height, padding_width=40, nn=None, reproduce_rate=0.05):
        self.stage = 0
        self.max_stage = 100
        self.screenshot_counter = 0 
        self.width = width
        self.height = height
        self.padding_width = padding_width
        self.position = pygame.math.Vector2(x, y)
        self.reproduce_rate = reproduce_rate
        self.max_speed = MAX_SPEED
        self.acceleration = ACCELERATION
        self.x_speed = 0
        self.y_speed = 0
        self.max_energy = MAX_ENERGY
        self.energy = START_ENERGY
        self.alive = True
        self.max_distance_per_frame = MAX_DISTANCE_PER_FRAME
        self.sex = random.choice([1, 2])
        # The new version will have Neural Network stored in the Animal class
        self.nn = nn
        if self.sex == 1:
            self.color = BLUE
        else:
            self.color = PINK
        self.weight = random.randint(1, 10)
        self.facing_direction = random.randint(0, 360)

    def land(self, x, y):
        self.position = pygame.math.Vector2(x, y)

    # This function accelerates the animal in the direction given
    def accelerate_towards(self, direction, dt):
        speed = math.sqrt(self.x_speed ** 2 + self.y_speed ** 2)
        # If the speed is already at the maximum speed, don't accelerate
        if speed >= self.max_speed:
            return
        if direction == 0:
            self.y_speed -= self.acceleration * dt

        elif direction == 1:
            self.y_speed += self.acceleration * dt

        elif direction == 2:
            self.x_speed -= self.acceleration * dt

        elif direction == 3:
            self.x_speed += self.acceleration * dt
        else:
            # If the direction is 4, don't accelerate and no energy is consumed
            return
        self.energy -= MOVEMENT_ENERGY_COST*dt

    # This function applies drag to the animal every frame
    def apply_drag(self, drag_coefficient, dt):
        if self.x_speed < 0.01 and self.x_speed > -0.01:
            self.x_speed = 0
        else:
            self.x_speed *= 1 - drag_coefficient * dt
        if self.y_speed < 0.01 and self.y_speed > -0.01:
            self.y_speed = 0
        else:
            self.y_speed *= 1 - drag_coefficient * dt
        

    # This function updates the position of the animal based on its speed
    def update_position(self, dt):
        new_x = self.position.x + self.x_speed * dt
        new_y = self.position.y + self.y_speed * dt
        if new_x < self.padding_width:
            self.die()
            new_x = self.padding_width
            self.x_speed = 0
        elif new_x > self.width + self.padding_width:
            self.die()
            new_x = self.width + self.padding_width
            self.x_speed = 0
        if new_y < self.padding_width:
            self.die()
            new_y = self.padding_width
            self.y_speed = 0
        elif new_y > self.height + self.padding_width:
            self.die()
            new_y = self.height + self.padding_width
            self.y_speed = 0
        distance = math.sqrt((new_x - self.position.x) ** 2 + (new_y - self.position.y) ** 2)
        if distance < self.max_distance_per_frame:
            self.position = pygame.math.Vector2(new_x, new_y)


    # update function that is called every frame
    def update(self, action, dt):
        if not self.alive or dt > LAG_THRESHOLD:
            return
        self.accelerate_towards(action, dt)
        self.update_position(dt)
        self.apply_drag(DRAG_COEFFICIENT, dt)
        self.energy -= ENERGY_COST_PER_FRAME*dt
        self.check_energy()

    def grow(self):
        if self.stage < self.max_stage:
            if self.energy / self.max_energy > 0.8:
                self.stage += 1
                self.max_energy += 10
                self.max_speed += 10
                self.acceleration -= 10
                self.energy -= self.max_energy * 0.5

    # draw the animal on the given surface
    def draw(self, surface):
        # Draw the body
        color = self.color
        if not self.alive:
            color = (50, 50, 50)
        pygame.draw.circle(surface, color, (int(self.position.x), int(self.position.y)), BASE_RADIUS+self.stage) 

    # draw the energy bar of the animal
    def draw_energy(self, surface,):
        # Draw the energy bar
        bar_width = 40
        bar_height = BASE_RADIUS + self.stage
        energy_ratio = self.energy / 100
        energy_bar_width = bar_width * energy_ratio
        bar_x = int(self.position.x - bar_width / 2)
        bar_y = int(self.position.y - 10)  # Position the bar above the animal
        # Draw the background of the energy bar (gray)
        pygame.draw.rect(surface, (128, 128, 128), (bar_x, bar_y, bar_width, bar_height))
        # Draw the foreground of the energy bar (green)
        pygame.draw.rect(surface, (0, 255, 0), (bar_x, bar_y, energy_bar_width, bar_height))

    def draw_name(self, surface, font, name):
        text = font.render(name, True, (0, 0, 0))
        surface.blit(text, (self.position.x - 10, self.position.y - 15 - BASE_RADIUS - self.stage))

    
    def hit (self, plant):
        distance = self.position.distance_to(pygame.math.Vector2(plant.position.x, plant.position.y))
        return distance < 10
    
    def eat(self, plant):
        if self.can_eat(plant) and plant.provide_fruit():
            self.energy += 20
            if self.energy > self.max_energy:
                self.energy = 100
            return True
        return False

    def can_eat(self, plant):
        distance = self.position.distance_to(pygame.math.Vector2(plant.position.x, plant.position.y))
        return distance < 10  # Threshold distance to eat the plant

    def check_energy(self):
        if self.energy <= 0:
            print("Animal died due to lack of energy")
            self.die()

    def die(self):
        self.alive = False

    def get_screenshot(self, screen, width, height):
        x, y = int(self.position.x), int(self.position.y)
        rect = pygame.Rect(x - width // 2, y - height // 2, width, height)
        
        screen_rect = screen.get_rect()
        rect.clamp_ip(screen_rect)
        
        screenshot_surface = screen.subsurface(rect).copy()
        screenshot = pygame.surfarray.array3d(screenshot_surface)
        screenshot = np.transpose(screenshot, (1, 0, 2))  # Transpose to match OpenCV's format
        
        return screenshot    
    
    def get_eat(self):
        if self.alive:
            self.die()
            return 1
        return 0
        
    def reproduce(self):
        random = np.random.rand()
        if random < self.reproduce_rate:
            return None
        else:
            if self.energy > 50:
                self.energy -= 50
                return Animal(self.position.x, self.position.y, self.width, self.height, self.padding_width, self.nn, self.reproduce_rate)
            return None
        
    
    