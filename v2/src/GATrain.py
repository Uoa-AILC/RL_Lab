
import torch
from AnimalEnvMultiAgent import AnimalEnv
from Buffer import RewardRecorder
from AnimalTrainer import GATrainer

NUM_AGENTS = 30
NUM_PLANTS = 50

MAX_STEPS = 3000000
MAX_EPSIDOE_STEPS = 2000

UPDATE_FREQUENCY = 3
SAVE_FREQUENCY = 300000

MUTATION_RATE = 1
MUTATION_NUMBER = 3
NUM_AGENTS_TO_SELECT = 5
REPRODUCE_METHOD = 'even'

SPEED_FACTOR = 6
DT_FACTOR = 0.1 
RENDER_MODE = "Single"

WINDOW_WIDTH = 600
WINDOW_HEIGHT = 600

image_shape = (20, 20, 3)
input_feature_size = 3
output_feature_size = 5



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    parallel_env = AnimalEnv(NUM_AGENTS, num_plants=NUM_PLANTS, window_height=WINDOW_HEIGHT, window_width=WINDOW_WIDTH, speed_factor=SPEED_FACTOR, image_shape=image_shape, dt_factor=DT_FACTOR, render_mode=RENDER_MODE)
    GA_trainer = GATrainer(image_shape, input_feature_size, output_feature_size, NUM_AGENTS, NUM_PLANTS, device, WINDOW_WIDTH, WINDOW_HEIGHT, False)

    # Load the model if it exists
    try:
        GA_trainer.load_model("GA_model_final.pt")
    except Exception as e:
        print(e)

    step = 0
    rewards_record = RewardRecorder()
    num_episodes = 0

    while step < MAX_STEPS:
        # Reset the environment
        current_state = parallel_env.reset(GA_trainer.population, GA_trainer.plants, GA_trainer.possible_agents)
        episode_step = 0
        num_episodes += 1

        # Run the environment for one episode
        while parallel_env.agents and episode_step < MAX_EPSIDOE_STEPS:
            actions = {}
            for agent in parallel_env.agents:
                actions[agent] = GA_trainer.get_action(current_state[agent], parallel_env.action_space(agent), agent)
            next_state, rewards, _, _, skip, agent_names = parallel_env.step(actions)
            # Skip the frame
            if skip:
                continue
            for agent_name in agent_names:
                # Add the reward to the reward recorder
                rewards_record.add_reward(agent_names[agent_name], rewards[agent_name])
                
            episode_step += 1
            step += 1
            if step % 1000 == 0:
                print(f"Step: {step}")
                fps = parallel_env.clock.get_fps()
                print(f"FPS: {fps}")
            # Save the model every SAVE_FREQUENCY steps
            if step % SAVE_FREQUENCY == 0:
                GA_trainer.save_model(f"GA_model_{step}.pt")

        # Evaluate the agents after each episode
        experience = rewards_record.get_rewards()
        GA_trainer.evaluate(experience)
        
        # Update the population every UPDATE_FREQUENCY episodes
        if num_episodes % UPDATE_FREQUENCY == 0:
            selected_agents = GA_trainer.select_population(NUM_AGENTS_TO_SELECT)
            GA_trainer.update_population(selected_agents, method=REPRODUCE_METHOD, mutation_rate=MUTATION_RATE, num_mutations=MUTATION_NUMBER)
    
    GA_trainer.save_model("GA_model_final.pt")



