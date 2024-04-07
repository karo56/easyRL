<div align="center">

#  🏆 👾 🎮 EasyRL library 🎮 👾  🏆
</div>

## 🚀 Quick introduction  🚀

What if we want to train RL models on some environment? 
We use stable-baseline3 library and gym and train! 
But what if we want to perform, multiple experiments, on multiple environments
on multiple models on multiple network architectures?

We can use the library/template **easyRL** ☺️

These repository use the stable-baseline3 library and it's functions to make simple
experimenting with ability to easy change models/architectures and RL environments and the hydra library to manage all configs.
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


_Note: You can use **TensorFlow** if you want to, but I am big fan of PyTorch and used only this library_

_Note 2: I did not add the usage of any library to track and store the results of experiments like [ClearML](https://clear.ml/) or [MLflow](https://mlflow.org/)
because many people use different ones, and adding such a library is very simple and anyone can do it themselves.
I encourage you to add your favorite._ 😊 
Instead of it I created dashboard based on [streamlit](https://streamlit.io/)  library. More information [below](#dashboard). 
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

```bash
bash bash/run_training.sh
```

## Structure of project

### Pipeline
TODO

### Description of files


This library have standard and simply structure. Diagram below explain what each file is.

It is worth mentioning that every experiment is saved in folder outputs with structure:
```outputs/experiments/<env_name>/<algo_type>/<ID_experiment_name>``` (this folder is created automatically)

```
├── bash                    <- All bash scripts
├── config                  <- Folder with all configs for Hydra
├── dashboard               <- Code for dashboard script to life tracking experiments
├── easyRL                  <- Source code of package 
├── outputs                 <- Folder contains all outputs of experiments 
├── sandbox                 <- Notebooks and some testing scripts
├── steps/training.py       <- main script which run RL experiment
```

Library easyRL contains several files:
```
├── easyRL               
│   ├── custom_wrappers          
│   │   ├── callbacks.py         <- Custom callbacks e.g. MakeGifCallback
│   │   ├── wrappers.py          <- Custom wrappers if you want to add own wrappers, write it here
│   ├── custom_policies         
│   │   ├── policies.py          <- Custom net architechtures written in Pytorch, you can add your own nets here
│   ├── prepare_env         
│   │   ├── prepare_env.py       <- Function which prepare env and add all used wrappers
│   ├── utils         
│   │   ├── utils.py             <- Some usefull functions to make code easier to read, for example code to create experiment folder
```

## Examples
Below you can find several examples, how to run different experiments and how to change it.

<details>
<summary><b>Change model </b></summary>

If you want to change model, you can simply change one line in `bash/run_training.sh` file:

```
python steps/training.py \
  experiment=lunar_lander_dqn \
  total_timesteps=20_000 \
  model=a2c  <-change this line, you can wirte all models in folder: config/model/.. e.g. ppo
```
_Note: you can be confused that we use experiment with contains dqn model, but we change it to a2c.
This is hydra package magic, if we run bash script we can override every param, so don't worry about it.
Experiment tells to hydra that we want to run dqn model on lunar lander env but model override change model._

</details>

<details>
<summary><b>Change params, for example learning rate </b></summary>

Exactly like in example below, we just add lines in bash script  `bash/run_training.sh`

```
python steps/training.py \
  experiment=lunar_lander_dqn \
  total_timesteps=20_000 \      <- this changes numer of timesteps
  model.buffer_size=10_000 \    <- we change bufer size in dqn
  model.learning_rate=0.0001 \  <- we change learning rate
  policy_net=custom_mlp         <- we change net architecture
```

We can also change it in `config/model` files, but please remember that you change it for every new run
not only for one experiment (but maybe this is what you want to do 😍)
</details>

<details>
<summary><b>Create new experiment </b></summary>
TODO
</details>


<details>
<summary><b>Run experiment without experiment config </b></summary>
If you don't like use experiments config or you just want to run simple experiment you don't have to use it.
Below examples :)
</details>


<details>
<summary><b>Change default value in config </b></summary>
TODO
</details>



<details>
<summary><b>Use CNN nets </b></summary>
Very large number of environments have representation by image. It is then worth using the CNN network, below are some examples of how to do it.
Auto-cnn.
TODO
</details>


## How use library for own environments and experiments?
This is probably the most important and interesting part of readme. 
How use easyRL package for your own environment/algorithm/net architecture? 
I hope this is super simply (if not, let me know! 😭) and below you can find several examples:

<details>
<summary><b>How to use easyRL for own envs?</b></summary>

only gym
TODO
</details>

<details>
<summary><b>How to add own wrappers?</b></summary>

TODO
</details>


<details>
<summary><b>How to add own RL algorithms like e.g. TRPO? </b></summary>
only stable-baseline3 but we have https://sb3-contrib.readthedocs.io/en/master/guide/install.html
</details>

<details>
<summary><b>How to add own net architectures? </b></summary>

TODO
</details>


<details>
<summary><b>How to manage configs and how to change it? </b></summary>
hydra and template

TODO
</details>

## 📊 📈  <a name="dashboard"></a> Tracking experiments 
RL experiments take a very long time, sometimes even many days/weeks. 
It is good to know how experiment perform and have estimation time when experiments end.
All these features have dashboard created with [streamlit](https://streamlit.io/) package. 
You can run this with simple bash command:

```bash
bash bash/run_dashboard.sh
```
Now, just go to http://localhost:8501 choose your env, algorithm and experiment to see how it performs.


_Note: envs, algorithms and experiments are based on outputs/experiments folder. If this folder is empty, you would see nothing special on this dashboard_ 😥 


TODO: you can also use tensorboard loger if you want :)

## 

##  F&Q
<details>
<summary><b>Why don't we use any library to track experiments? </b></summary>

As I wrote above, for the sake of simplifying the library, I did not add such a library. Everyone has their own favorite so I encourage you to add according to your needs!
</details>

<details>
<summary><b>Why prepare_env function have default render_mode = "rgb_array" </b></summary>

In older version of Gym library we could choose render model when we made `env.render(mode=xxx)`. 
Now we have to define render mode when env is inited, so we should use "rgb_array" because of `MakeGifCallback`.
</details>


<details>
<summary><b>How to track experiments? </b></summary>
TODO
</details>

<details>
<summary><b>TODO </b></summary>
TODO
</details>

## References
If you want to read some good repos, those are my inspirations:
  - https://github.com/krystianfranus/data-science-template
  - https://github.com/ashleve/lightning-hydra-template
