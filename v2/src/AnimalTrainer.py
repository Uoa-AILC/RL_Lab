import json
import random
import torch
import math
import copy
from animal_V2 import Animal
from SimpleNN import MiniNN

from plant import Plant

WINDOW_WIDTH = 600
WINDOW_HEIGHT = 600

# The class that defines the Genetic Algorithm Trainer
class GATrainer:
    def  __init__(self, image_shape, input_feature_size, output_feature_size, agent_size=15, plant_size=25, device='cpu', window_width=WINDOW_WIDTH, window_height=WINDOW_HEIGHT, is_simple=False, is_reproducing=False):
        self.agent_size = agent_size
        self.plant_size = plant_size
        self.window_width = window_width
        self.window_height = window_height
        self.population = {}
        self.performance = {}
        self.best_model = None
        self.image_shape = image_shape
        self.input_feature_size = input_feature_size
        self.output_feature_size = output_feature_size
        self.device = device
        self.is_simple = is_simple
        self.is_reproducing = is_reproducing
        # create a list of possible agents                                                       
        self.possible_agents = ["agent_" + str(r) for r in range(self.agent_size)]
        # create the plants
        self.plants = [Plant((0, 255, 0), (255, 0, 0)) for _ in range(plant_size)]
        for agent_name in self.possible_agents:
            self.add_agent(agent_name)

    def __len__(self):
        return len(self.population)
    
    def get_new_animal(self, nn):
        return Animal(0, 0, self.window_width, self.window_height, 40, nn, 0.05)

    # funtion to get a simple neural network instance
    def get_new_nn(self):
        return MiniNN(math.prod(self.image_shape)+self.input_feature_size, self.output_feature_size).to(self.device)

    # function to add an agent to the population
    def add_agent(self, agent_name):
        nn = self.get_new_nn()
        animal = self.get_new_animal(nn)
        self.population[agent_name] = animal
        print(f"New agent added: {agent_name}")

    # function to get the action of an agent
    def get_action(self, agent_state, action_space, agent_name):
        if agent_name not in self.population:
            return action_space.sample()
        else:
            # Exploitation: Model-predicted best action
            with torch.no_grad():
                state_tensor = torch.tensor(agent_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_values = self.population[agent_name].nn(state_tensor)
                return torch.argmax(q_values).item()

    # function to evaluate the performance of the agents and update the performance dictionary
    def evaluate(self, rewards):
        # normalize rewards
        max_abs_reward = max(abs(r) for r in rewards.values())
        for agent_name in rewards:
            if agent_name not in self.population:
                continue
            normalized_reward = rewards[agent_name] / max_abs_reward
            self.performance[agent_name] = self.performance.get(agent_name, 0) + normalized_reward

    # function to select the top k agents
    def select_population(self, top_k=5):
        sorted_agents = sorted(self.performance.items(), key=lambda x: x[1], reverse=True)
        selected_agents = [agent for agent, _ in sorted_agents[:top_k]]
        print(f"Selected agents: {selected_agents}")
        return selected_agents

    # function to update the population by reproducing the selected agents and removing the rest
    def update_population(self, selected_agents, method="rand", mutation_rate=1, num_mutations=4):
        self.performance = {agent: 0 for agent in self.performance}
        for agent in list(self.population.keys()):
            if agent not in selected_agents:
                self.remove_agent(agent)

        num_children = self.agent_size - len(self.population)
        print(f"Number of children: {num_children}")
        for i in range(num_children):
            parent1 = self.population[selected_agents[i % len(selected_agents)]].nn
            parent2 = self.population[selected_agents[(i + 1) % len(selected_agents)]].nn
            child_nn = self.reproduce_two(parent1, parent2, method)
            child_animal = self.get_new_animal(child_nn)
            for agent_name in self.possible_agents:
                if agent_name not in self.population:
                    self.population[agent_name] = child_animal
                    #print(f"New agent produced {agent_name}")
                    self.mutate_agent(mutation_rate, agent_name, num_mutations=num_mutations)
                    break

    def reproduce_two(self, parent1, parent2, method="even"):
        child = self.get_new_nn()
        with torch.no_grad():
            for child_param, p1_param, p2_param in zip(child.parameters(), parent1.parameters(), parent2.parameters()):
                if method == "even":
                    # Evenly mix
                    child_param.data.copy_((p1_param.data + p2_param.data) / 2)
                else:
                    # Randomly choose each parameter element
                    mask = torch.rand_like(p1_param) < 0.5
                    child_param.data.copy_(torch.where(mask, p1_param.data, p2_param.data))
        return child

    def mutate_agent(self, degree, agent_name, num_mutations=2):
        with torch.no_grad():
            params = list(self.population[agent_name].nn.parameters())
            selected_params = random.sample(params, min(num_mutations, len(params)))
            for param in selected_params:
                param.add_(torch.randn_like(param) * degree)


    def remove_agent(self, agent_name):
        self.population.pop(agent_name)
        self.performance.pop(agent_name)
        #print(f"Agent {agent_name} removed")
        
    # def expand_population(self):
    #     loaded_agents = list(self.population.keys())
    #     for i in range(self.max_population_size):
    #         agent_name = f"agent_{i}"
    #         if agent_name not in loaded_agents:
    #             parent1 = random.choice(loaded_agents)
    #             parent2 = random.choice(loaded_agents)
    #             child = self.reproduce_two(self.population[parent1], self.population[parent2])
    #             self.population[agent_name] = child
    #             self.population_size += 1
    #             self.mutate_agent(1, agent_name, num_mutations=2)
    #             print(f"New agent produced {agent_name}")

    def load_model(self, filename):
        checkpoint = torch.load(filename, weights_only=True)
        num_loaded = 0
        for idx, state_dict in enumerate(checkpoint['models']):
            agent_name = f"agent_{idx}"
            nn = self.get_new_nn()
            nn.load_state_dict(state_dict)
            new_animal = self.get_new_animal(nn)
            self.population[agent_name] = new_animal
            print(f"Agent {agent_name} loaded")
            self.performance[agent_name] = 0
            num_loaded += 1
            if num_loaded >= self.agent_size:
                break
        self.population_size += len(checkpoint['models'])
        print(f"Population size: {self.population_size}")

    def save_model(self, filename):
        selected_agents = self.select_population()
        agent_models = [self.population[agent].nn for agent in selected_agents]
        states = [model.state_dict() for model in agent_models]
        torch.save({'models': states}, filename)

    # def load_model_json(self, filename):
    #     with open(filename, 'r') as f:
    #         states = json.load(f)
    #     num_loaded = 0
    #     for idx, state in enumerate(states):
    #         agent_name = f"agent_{idx}"
    #         nn = self.get_simple_nn()
    #         nn.load_state_dict({k: torch.tensor(v, device=self.device) for k, v in state.items()})
    #         self.population[agent_name] = nn
    #         print(f"Agent {agent_name} loaded")
    #         self.performance[agent_name] = 0
    #         num_loaded += 1
    #         if num_loaded >= self.max_population_size:
    #             break
    #     self.population_size += len(states)
    #     print(f"Population size: {self.population_size}")


    # def save_model_json(self, filename):
    #     selected_agents = self.select_population()
    #     agent_models = [self.population[agent] for agent in selected_agents]
    #     states = [model.state_dict() for model in agent_models]
    #     serializable_states = []
    #     for state in states:
    #         serializable_state = {}
    #         for key, tensor in state.items():
    #             serializable_state[key] = tensor.cpu().tolist()
    #         serializable_states.append(serializable_state)
    #     with open(filename, 'w') as f:
    #         json.dump(serializable_states, f, indent=2)
    #     print(f"Models saved to {filename}")

    # def save_best_model(self, filename):
    #     selected_agents = self.select_population()
    #     self.best_model = selected_agents[0]
    #     state = self.population[self.best_model].state_dict()
    #     serializable_state = {}
    #     for key, tensor in state.items():
    #         serializable_state[key] = tensor.cpu().tolist()
    #     with open(filename, 'w') as f:
    #         json.dump(serializable_state, f, indent=2)
    #     print(f"Best model saved to {filename}")

class NSTrainer:
    def  __init__(self, image_shape, input_feature_size, output_feature_size, max_agent_size=15, plant_size=25, device='cpu', window_width=WINDOW_WIDTH, window_height=WINDOW_HEIGHT, is_simple=False, is_reproducing=False):
        self.max_agent_size = max_agent_size
        self.plant_size = plant_size
        self.window_width = window_width
        self.window_height = window_height
        self.population = {}
        self.image_shape = image_shape
        self.input_feature_size = input_feature_size
        self.output_feature_size = output_feature_size
        self.device = device
        self.is_reproducing = is_reproducing
        # create a list of possible agents                                                       
        self.possible_agents = ["agent_" + str(r) for r in range(self.max_agent_size)]
        # create the plants
        self.plants = [Plant((0, 255, 0), (255, 0, 0)) for _ in range(plant_size)]
        for agent_name in self.possible_agents:
            self.add_agent(agent_name)

    def __len__(self):
        return len(self.population)

    def get_new_animal(self, nn):
        return Animal(0, 0, self.window_width, self.window_height, 40, nn, 0.05)

    # funtion to get a simple neural network instance
    def get_new_nn(self):
        return MiniNN(math.prod(self.image_shape)+self.input_feature_size, self.output_feature_size).to(self.device)

    # function to add an agent to the population
    def add_agent(self, agent_name):
        nn = self.get_new_nn()
        animal = self.get_new_animal(nn)
        self.population[agent_name] = animal
        print(f"New agent added: {agent_name}")

    # function to get the action of an agent
    def get_action(self, agent_state, action_space, agent_name):
        if agent_name not in self.population:
            return action_space.sample()
        else:
            # Exploitation: Model-predicted best action
            with torch.no_grad():
                state_tensor = torch.tensor(agent_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_values = self.population[agent_name].nn(state_tensor)
                return torch.argmax(q_values).item()

    def add_new_born(self, agent):
        if (len(self.population) < self.max_agent_size):
            for agent_name in self.possible_agents:
                if agent_name not in self.population:
                    self.population[agent_name] = agent
                    agent.nn = copy(agent.nn)
                    self.mutate_agent(1, agent_name, num_mutations=2)
                    print(f"New agent produced {agent_name}")
            else:
                print("Population is full")

    def mutate_agent(self, degree, agent_name, num_mutations=2):
        with torch.no_grad():
            params = list(self.population[agent_name].nn.parameters())
            selected_params = random.sample(params, min(num_mutations, len(params)))
            for param in selected_params:
                param.add_(torch.randn_like(param) * degree)

    def reproduce_two(self, parent1, parent2, method="even"):
        child = self.get_new_nn()
        with torch.no_grad():
            for child_param, p1_param, p2_param in zip(child.parameters(), parent1.parameters(), parent2.parameters()):
                if method == "even":
                    # Evenly mix
                    child_param.data.copy_((p1_param.data + p2_param.data) / 2)
                else:
                    # Randomly choose each parameter element
                    mask = torch.rand_like(p1_param) < 0.5
                    child_param.data.copy_(torch.where(mask, p1_param.data, p2_param.data))
        return child
    
    def load_model(self, filename):
        checkpoint = torch.load(filename, weights_only=True)
        num_loaded = 0
        for idx, state_dict in enumerate(checkpoint['models']):
            agent_name = f"agent_{idx}"
            nn = self.get_new_nn()
            nn.load_state_dict(state_dict)
            new_animal = self.get_new_animal(nn)
            self.population[agent_name] = new_animal
            print(f"Agent {agent_name} loaded")
            self.performance[agent_name] = 0
            num_loaded += 1
            if num_loaded >= self.agent_size:
                break
        self.population_size += len(checkpoint['models'])
        print(f"Population size: {self.population_size}")

    def save_model(self, filename):
        selected_agents = self.select_population()
        agent_models = [self.population[agent].nn for agent in selected_agents]
        states = [model.state_dict() for model in agent_models]
        torch.save({'models': states}, filename)

