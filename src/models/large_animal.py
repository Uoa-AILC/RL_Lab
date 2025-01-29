import math
import random
import pygame
import numpy as np

LAG_THRESHOLD = 0.2
class Animal:
    def __init__(self, x, y, width, height, color, padding_width=40):
        self.screenshot_counter = 0 
        self.width = width
        self.height = height
        self.padding_width = padding_width
        self.position = pygame.math.Vector2(x, y)
        self.color = color
        self.max_speed = 120
        self.acceleration = 70
        self.x_speed = 0
        self.y_speed = 0
        self.angular_velocity = 0.1
        # self.target = None
        self.max_energy = 200
        self.energy = 200
        self.alive = True
        self.max_distance_per_frame = 1000
        self.sex = random.choice([1, 2])
        self.weight = random.randint(1, 10)
        self.facing_direction = random.randint(0, 360)

    def accelerate_towards(self, direction, dt):
        speed = math.sqrt(self.x_speed ** 2 + self.y_speed ** 2)
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
            return
        self.energy -= 0.8*dt


    def apply_drag(self, drag_coefficient, dt):
        if self.x_speed < 0.01 and self.x_speed > -0.01:
            self.x_speed = 0
        else:
            self.x_speed *= 1 - drag_coefficient * dt
        if self.y_speed < 0.01 and self.y_speed > -0.01:
            self.y_speed = 0
        else:
            self.y_speed *= 1 - drag_coefficient * dt
        

        
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

    def update(self, action, dt):
        if not self.alive or dt > LAG_THRESHOLD:
            return
        self.accelerate_towards(action, dt)
        self.update_position(dt)
        self.apply_drag(0.98, dt)
        self.energy -= 0.2*dt
        self.check_energy()

    def draw(self, surface):
        # Draw the body
        color = self.color
        if not self.alive:
            color = (50, 50, 50)
        pygame.draw.circle(surface, color, (int(self.position.x), int(self.position.y)), 5)

    def draw_energy(self, surface,):
        # Draw the energy bar
        bar_width = 40
        bar_height = 5
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
        surface.blit(text, (self.position.x - 10, self.position.y - 20))

    def hit (self, plant):
        distance = self.position.distance_to(pygame.math.Vector2(plant.position.x, plant.position.y))
        return distance < 10
    
    def eat(self, animal):
        if self.can_eat(animal):
            if animal.get_eat():
                self.energy += 100
                if self.energy > self.max_energy:
                    self.energy = self.max_energy
            return True
        return False

    def can_eat(self, animal):
        distance = self.position.distance_to(pygame.math.Vector2(animal.position.x, animal.position.y))
        return distance < 10  # Threshold distance to eat the plant

    def check_energy(self):
        if self.energy <= 0:
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
        
        # if self.screenshot_counter < 5:
        #     cv2.imshow('Screenshot', cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR))
        #     cv2.waitKey(0)  # Wait for a key press to close the window
        #     cv2.destroyAllWindows()
        #     self.screenshot_counter += 1  # Increment the counter

        return screenshot    