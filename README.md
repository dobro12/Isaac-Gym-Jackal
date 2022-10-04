# Isaac-Gym-Jackal

## Prerequisite

1. Isaac Gym 1.0.preview4
2. https://github.com/NVIDIA-Omniverse/IsaacGymEnvs

## Environment Explanation
- To perceive obstacles, image information is given to RL agents.
- We preprocess the image information uisng VAE, and the encoded vectors are given as observations.
- Therefore, the environment is constructed as follows:
    1. Images are collected in `vae/collect.py` with the `DummyJackal` task.
    2. Encder and Decoder are trained in `vae/main.py`.
    3. Using the trained encoder, the `Jackal` task is constructed in `utils/jackal_env/task2.py`.


## How to train RL agents
```bash
# 1. Collect images.
cd vae
python collect.py
# 2. Train VAE.
python main.py
# 3. Train PPO agents.
cd ../ppo
python main.py
# 4. Visualize PPO agents.
python main.py --test
```