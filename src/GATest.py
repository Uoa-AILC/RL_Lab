
import torch
from env.AnimalEnvMultiAgent import AnimalEnv
from models.Buffer import RewardRecorder
from GA.trainer.AnimalTrainer import GATrainer

NUM_AGENTS = 10
MAX_STEPS = 1000000
UPDATE_FREQUENCY = 2
MAX_EPSIDOE_STEPS = 5000
SAVE_FREQUENCY = 100000
NUM_PLANTS = 20
MUTATION_RATE = 0.2
NUM_AGENTS_TO_SELECT = 5
REPRODUCE_METHOD = 'rand'
SPEED_FACTOR = 3
DT_FACTOR = 0.1

image_shape = (32, 32, 3)
input_feature_size = 3
output_feature_size = 5



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parallel_env = AnimalEnv(NUM_AGENTS, num_plants=NUM_PLANTS, speed_factor=SPEED_FACTOR, dt_factor=DT_FACTOR)
    GA_trainer = GATrainer(image_shape, input_feature_size, output_feature_size, NUM_AGENTS, device, False)
    try:
        GA_trainer.load_model("GA_model_final.pt")
    except Exception as e:
        print(e)
        for agent in parallel_env.possible_agents:
            GA_trainer.add_agent(agent)

    step = 0
    rewards_record = RewardRecorder()
    num_episodes = 0

    while step < MAX_STEPS:
        current_state, infos = parallel_env.reset()
        episode_step = 0
        num_episodes += 1
        while parallel_env.agents and episode_step < MAX_EPSIDOE_STEPS:
            actions = {}
            for agent in parallel_env.agents:
                actions[agent] = GA_trainer.get_action(current_state[agent], parallel_env.action_space(agent), 0.1, agent)
            next_state, rewards, terminations, truncations, skip, agent_names = parallel_env.step(actions)
            if skip:
                continue
            for agent_name in agent_names:
                rewards_record.add_reward(agent_names[agent_name], rewards[agent_name])
            current_state = next_state
            episode_step += 1
            step += 1
            if step % 1000 == 0:
                print(f"Step: {step}")
        experience = rewards_record.get_rewards()
        GA_trainer.evaluate(experience)
        if num_episodes % UPDATE_FREQUENCY == 0:
            selected_agents = GA_trainer.select_population(NUM_AGENTS_TO_SELECT)
            GA_trainer.update_population(selected_agents, parallel_env.possible_agents, method=REPRODUCE_METHOD, mutation_rate=MUTATION_RATE)


