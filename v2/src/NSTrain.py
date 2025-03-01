
import torch
from AnimalEnvMultiAgent import AnimalEnv
from Buffer import RewardRecorder
from AnimalTrainer import NSTrainer

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

SPEED_FACTOR = 10
DT_FACTOR = 0.1 
RENDER_MODE = "Human"

WINDOW_WIDTH = 600
WINDOW_HEIGHT = 600

image_shape = (32, 32, 3)
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
    trainer = NSTrainer(image_shape, input_feature_size, output_feature_size, NUM_AGENTS, NUM_PLANTS, device, WINDOW_WIDTH, WINDOW_HEIGHT, False)

    # Load the model if it exists
    try:
        trainer.load_model("GA_model_final.pt")
    except Exception as e:
        print(e)

    step = 0
    rewards_record = RewardRecorder()
    num_episodes = 0
    current_state = parallel_env.reset(trainer.population, trainer.plants, trainer.possible_agents)

    while step < MAX_STEPS:
        actions = {}
        for agent in parallel_env.agents:
            actions[agent] = trainer.get_action(current_state[agent], parallel_env.action_space(agent), agent)
        next_state, rewards, _, _, skip, agent_names = parallel_env.step(actions)
        current_state = next_state
        # Skip the frame
        if skip:
            continue
        for agent in parallel_env.agent_instances:
            child = parallel_env.agent_instances[agent].reproduce()
            trainer.add_new_born(child)


        step += 1
        if step % 1000 == 0:
            print(f"Step: {step}")
            fps = parallel_env.clock.get_fps()
            print(f"FPS: {fps}")
        # Save the model every SAVE_FREQUENCY steps
        if step % SAVE_FREQUENCY == 0:
            trainer.save_model(f"NS_model_{step}.pt")
        
        # Update the population every UPDATE_FREQUENCY episodes
    
    trainer.save_model("NS_model_final.pt")



