import random
import numpy as np
import torch
import math
from ...models.SimpleNN import SimpleNN



class GATrainer:
    def  __init__(self, image_shape, input_feature_size, output_feature_size, max_population_size=15, device='cpu', is_simple=False):
        self.max_population_size = max_population_size
        self.population_size = 0
        self.population = {}
        self.performance = {}
        self.best_model = None
        self.image_shape = image_shape
        self.input_feature_size = input_feature_size
        self.output_feature_size = output_feature_size
        self.device = device
        self.is_simple = is_simple
        
    def get_simple_nn(self):
        return SimpleNN(math.prod(self.image_shape)+self.input_feature_size, self.output_feature_size).to(self.device)
        
    def add_agent(self, agent_name):
        self.population[agent_name] = self.get_simple_nn()
        self.population_size += 1
        self.performance[agent_name] = 0
        print(f"New agent added: {agent_name}")

    def get_action(self, agent_state, action_space, epsilon, agent_name):
        if agent_name not in self.population:
            return action_space.sample()
        else:
            # Exploitation: Model-predicted best action
            with torch.no_grad():
                state_tensor = torch.tensor(agent_state, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_values = self.population[agent_name](state_tensor)
                return torch.argmax(q_values).item()

    def evaluate(self, rewards):
        # normalize rewards
        max_abs_reward = max(abs(r) for r in rewards.values())
        for agent_name in rewards:
            if agent_name not in self.population:
                continue
            normalized_reward = rewards[agent_name] / max_abs_reward
            self.performance[agent_name] = self.performance.get(agent_name, 0) + normalized_reward

    def select_population(self, top_k=5):
        sorted_agents = sorted(self.performance.items(), key=lambda x: x[1], reverse=True)
        selected_agents = [agent for agent, _ in sorted_agents[:top_k]]
        print(f"Selected agents: {selected_agents}")   
        return selected_agents

    def update_population(self, selected_agents, possiable_names, method="rand", mutation_rate=1, num_mutations=4):

        self.performance = {agent: 0 for agent in self.performance}
        for agent in list(self.population.keys()):
            if agent not in selected_agents:
                self.remove_agent(agent)

        num_children = self.max_population_size - len(self.population)
        for i in range(num_children):
            parent1 = self.population[selected_agents[i % len(selected_agents)]]
            parent2 = self.population[selected_agents[(i + 1) % len(selected_agents)]]
            child = self.reproduce_two(parent1, parent2, method)
            for agent_name in possiable_names:
                if agent_name not in self.population:
                    self.population[agent_name] = child
                    self.population_size += 1
                    #print(f"New agent produced {agent_name}")
                    self.mutate_agent(mutation_rate, agent_name, num_mutations=num_mutations)
                    break
                    
    def reproduce_two(self, parent1, parent2, method="even"):
        child = self.get_simple_nn()
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
    
    # mix all selected agents to produce new agents
    def reproduce_all(self, method="even", selected_agents=None):
        if selected_agents is None:
            selected_agents = list(self.population.keys())
        child = self.get_simple_nn()
        for i in range(len(selected_agents)):
            parent = self.population[selected_agents[i]]
            with torch.no_grad():
                for child_param, parent_param in zip(child.parameters(), parent.parameters()):
                    if method == "even":
                        # Evenly mix
                        child_param.data.copy_((child_param.data + parent_param.data) / 2)
                    else:
                        # Randomly choose each parameter element
                        mask = torch.rand_like(child_param) < 0.5
                        child_param.data.copy_(torch.where(mask, child_param.data, parent_param.data))
        return child
    
    def mutate_specific_layer(self, degree):
        for model in self.population:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if "fc1" in name:  # Mutate only parameters in layer "fc1"
                        param.add_(torch.randn_like(param) * degree)

    def mutate_all(self, degree):
        for model in self.population:
            with torch.no_grad():
                for param in model.parameters():
                    param.add_(torch.randn_like(param) * degree)

    def mutate_agent(self, degree, agent_name, num_mutations=2):
        with torch.no_grad():
            params = list(self.population[agent_name].parameters())
            selected_params = random.sample(params, min(num_mutations, len(params)))
            for param in selected_params:
                param.add_(torch.randn_like(param) * degree)


    def remove_agent(self, agent_name):
        self.population.pop(agent_name)
        self.performance.pop(agent_name)
        self.population_size -= 1
        #print(f"Agent {agent_name} removed")

    def load_model(self, filename):
        checkpoint = torch.load(filename, weights_only=True)
        num_loaded = 0
        for idx, state_dict in enumerate(checkpoint['models']):
            agent_name = f"agent_{idx}"
            nn = self.get_simple_nn()
            nn.load_state_dict(state_dict)
            self.population[agent_name] = nn
            print(f"Agent {agent_name} loaded")
            self.performance[agent_name] = 0
            num_loaded += 1
            if num_loaded >= self.max_population_size:
                break
        self.population_size += len(checkpoint['models'])
        print(f"Population size: {self.population_size}")

    def save_model(self, filename):
        selected_agents = self.select_population()
        agent_models = [self.population[agent] for agent in selected_agents]
        states = [model.state_dict() for model in agent_models]
        torch.save({'models': states}, filename)
