<div align="center">

#  🏆 👾 🎮 EasyRL library 🎮 👾  🏆
</div>

## 🚀 Quick introduction  🚀

What if we want to train RL models on some environment? 
We use stable-baseline3 library and gym and train! 
But what if we want to perform, multiple experiments, on multiple environments
on multiple models on multiple network architectures?

We can use the library/template **easyRL** ☺️

These repository use the stable-baseline3 library and it's functions to create a convenient
experimenting with ability to easy change models/architectures and RL environments, and the hydra library to manage all configs.
If you have your own RL environment and want to train any net on it, instead of writing everything from scratch, you can use this code.

The main advantages of the library:

- tracking live how experiment perform
- easy to change models/params/envs
- every experiment is saved in separate folder
- ability to save model and gif how algo works every n_steps
- just simplicity :)

The easyRL library is the result of my master's thesis, which I had to face the challenges above.

If anyone has ideas on how to improve this code, I would appreciate any comments 💖


## Main libraries used
- [gymnasium](https://gymnasium.farama.org/) - this is new version of Gym. Library with various environments on which we can train models e.g. [Atari Breakout game](https://gymnasium.farama.org/environments/atari/breakout/)  
- [stable-baseline3](https://stable-baselines3.readthedocs.io/en/master/) - framework for  training reinforcement learning agents with implementations a lot of popular RL models e.g. DQN, PPO
- [hydra](https://hydra.cc/) - library to manage all configs (see dir: `config/config.yaml`) 
- [PyTorch](https://pytorch.org/) - used for custom CNN and MLP net architectures  (see dir: `easyRL/custom_policies/policies.py`)


_Note: You can use **TensorFlow** if you want to, but i am big fan of PyTorch and use only this library_

_Note 2: I did not add the use of any library to track and store the results of experiments like [ClearML](https://clear.ml/) or [MLflow](https://mlflow.org/)
because many people use different ones, and adding such a library is very simple and anyone can do it themselves.
I encourage you to add your favorite._
## Installation
To install, simply run the command below instead of _<your_env_name>_ specify your own environment name.
```yaml
conda create -n <your_env_name> python=3.11.8
conda activate <your_env_name>
pip install -e .
```

## 💻 Machine configuration 
Code used on ubuntu >= **22.04** \
I have no idea if the code will work on windows, but I encourage you **not** to check it 😊


## 🚀 Quickstart 🚀
To start the training simply run the example script from the bash folder.
This script training _DQN_ model on _LunarLander-v2_ environment.

Run from root folder with:

```
bash bash/run_training.sh
```

## Structure of project
TODO

## How use library for own environments and experiments?
TODO: library makes folder

## F&Q
<details>
<summary><b>why don't we use any library to track experiments? </b></summary>
As I wrote above, for the sake of simplifying the library, I did not add such a library. Everyone has their own favorite so I encourage you to add according to your needs!
</details>

## References
If you want to read some good repos, those are my inspirations:
  - https://github.com/krystianfranus/data-science-template
  - https://github.com/ashleve/lightning-hydra-template
