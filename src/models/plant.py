import pygame
import random


class Plant:
    def __init__(self, x, y, color, fruit_color):
        self.position = pygame.math.Vector2(x, y)
        self.growth_stage = 0
        self.max_growth_stage = 5
        self.fruit = 0
        self.max_fruit = 3
        self.color = color
        self.fruit_color = fruit_color

    def grow(self):
        if self.growth_stage < self.max_growth_stage:
            self.growth_stage += 0.01
        elif self.fruit < self.max_fruit:
            p = random.random()
            if p < 0.004:
                self.fruit += 1

    def draw(self, surface):
        # Draw the plant
        pygame.draw.circle(surface, self.color, (int(self.position.x), int(self.position.y)), 5)
        # Draw the fruit
        for i in range(self.fruit.__int__()):
            pygame.draw.circle(surface, self.fruit_color, (int(self.position.x) + (i * 10), int(self.position.y) - 10), 3)

    def provide_fruit(self):
        if self.fruit > 0:
            self.fruit -= 1
            return True
        return False