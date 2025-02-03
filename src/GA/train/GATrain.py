
import torch
from ...env.AnimalEnvMultiAgent import AnimalEnv
from ...models.Buffer import RewardRecorder
from ..trainer.AnimalTrainer import GATrainer

NUM_AGENTS = 40
NUM_PLANTS = 50

MAX_STEPS = 3000000
MAX_EPSIDOE_STEPS = 4000

UPDATE_FREQUENCY = 3
SAVE_FREQUENCY = 400000

MUTATION_RATE = 1
MUTATION_NUMBER = 6
NUM_AGENTS_TO_SELECT = 5
REPRODUCE_METHOD = 'rand'

SPEED_FACTOR =3
DT_FACTOR = 0.1
RENDER_MODE = "Single"


image_shape = (32, 32, 3)
input_feature_size = 3
output_feature_size = 5



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parallel_env = AnimalEnv(NUM_AGENTS, num_plants=NUM_PLANTS, speed_factor=SPEED_FACTOR, dt_factor=DT_FACTOR, render_mode=RENDER_MODE)
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
                fps = parallel_env.clock.get_fps()
                print(f"FPS: {fps}")

            if step % SAVE_FREQUENCY == 0:
                GA_trainer.save_model(f"GA_model_{step}.pt")
        experience = rewards_record.get_rewards()
        GA_trainer.evaluate(experience)
        if num_episodes % UPDATE_FREQUENCY == 0:
            selected_agents = GA_trainer.select_population(NUM_AGENTS_TO_SELECT)
            GA_trainer.update_population(selected_agents, parallel_env.possible_agents, method=REPRODUCE_METHOD, mutation_rate=MUTATION_RATE, num_mutations=MUTATION_NUMBER)
    
    GA_trainer.save_model("GA_model_final.pt")



