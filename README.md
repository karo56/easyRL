[//]: # (![GitHub Logo]&#40;https://github.com/karo56/easyRL/blob/images/logo.png&#41;)
<div align="center">
<img src="https://github.com/karo56/easyRL/blob/images/logo.png" alt="GitHub Logo" width="300" height="300">

# 

[![Static Badge](https://img.shields.io/badge/python%203.11-blue?style=for-the-badge&logo=python&logoColor=white&color=blue)](https://www.python.org/)
[![Static Badge](https://img.shields.io/badge/gymnasium-black?style=for-the-badge&link=https%3A%2F%2Fgymnasium.farama.org%2F)](https://gymnasium.farama.org/)
[![Static Badge](https://img.shields.io/badge/stable--baseline3-white?style=for-the-badge)](https://stable-baselines3.readthedocs.io/en/master/)
[![Static Badge](https://img.shields.io/badge/hydra-sky?style=for-the-badge&labelColor=33b8ff&color=33b8ff)](https://hydra.cc/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)

![Static Badge](https://img.shields.io/badge/licence-mit-%20%23ffa833?style=for-the-badge&labelColor=grey)



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
To install, firstly clone repo with code:
```yaml
git clone https://github.com/karo56/easyRL.git
cd easyRL
```

After that, simply run the command below instead of _<your_env_name>_ specify your own environment name.
```yaml
conda create -n <your_env_name> python=3.11.8
conda activate <your_env_name>
pip install -e .
```
Now, installation is done and you can use this library. 🎉🎉

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

## 📄 Structure of project

### Pipeline
To run every RL experiment we should have structured pipeline.
For easyRL we propose the following pipeline structure:
![](https://github.com/karo56/easyRL/blob/images/pipeline.png)

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

You can create experiment config file (see examples in dir: ```config/experiment/..```).
In this file you can write configuration on specific experiment. 
For example, you want to check how _PPO_ algo with specific parameters works on _PongNoFrameskip-v4_ environment.
Of course, you can write everything in argparse arguments, but you can also write simple experiment file and use it.
This allows you to save this experiment and its specifications. This is nice if you want to reproduce experiment later.

Let's see examples!

1) Experiment 1

We run PPO algo on env ```PongNoFrameskip-v4``` with ```batch_size = 128``` and ```learning_rate = 000.1```

We add everything to file ```pong_with_cnn_poo.yaml```:
```yaml
defaults:
  - override /env: pong_no_frame_skip
  - override /model: ppo
  - override /policy_net: default

model:
  policy: "CnnPolicy"
  batch_size: 128
  learning_rate: 000.1

experiment_name: "pong_with_cnn_poo"
```

2) Experiment 2 

Now, we create config params file ```mountain_car_a2c.yaml```:
```yaml
TODO
```

You switch experiments and params just change one line in file ```bash/run_training.sh``` file

Now, if we want to switch experiments we just change one line of code 
before run ```bash/run_training.sh``` script

1) Run experiment 1
```yaml
python steps/training.py \
  experiment=pong_with_cnn_poo \ <- change this line
  total_timesteps=20_000 
```

2) Run experiment 2
```yaml
python steps/training.py \
  experiment=lunar_lander_dqn \ <- change this line
  total_timesteps=20_000 
```
Easy(RL)? Yes. 😍 

</details>


<details>
<summary><b>Run experiment without experiment config </b></summary>

If you don't like use experiments config or you just want to run simple experiment you don't have to use it.
But remeber: you have to define model and environment! (There is not default model and enviroment :))

**Let's see example:**

We have to define env and model parameters in argparse. 
So for env parameters we choose one from yaml files from ```config/env/..``` folder and 
for model  we choose one from ```config/model/..``` folder.
(without the _.yaml_ suffix ❗)

```yaml
python steps/training.py \
  env=bipedal_walker \
  model=ppo \
  total_timesteps=20_000
```

</details>


<details>
<summary><b>Change default value in config </b></summary>

If you want to change some defult parameters to when you run every experiment is super simple.
Just go to file ```config/config.yaml``` file and change it. For example default value _total_timesteps_
from 50_000 into 1_000_000:

```yaml
# @package _global_
defaults:
  - _self_
  - env: null
  - model: null
  - policy_net: default

  - experiment: null

  - override hydra/job_logging: custom

hydra:
  output_subdir: null
  run:
    dir: .


# number of envs to train model
n_envs: 4

# The total number of samples (env steps) to train on
total_timesteps: 1_000_000   💥 <- we had 50_000 💥

# The number of episodes before logging.
log_interval: 1_000

# eval_frequency to save model and make gif, eval_frequency have to be >= 1
eval_frequency: 2

# dir of storage all experiments
path_to_outputs: "outputs/experiments"

# description of experiment
description: "This is experiment description, we can write whatever we want"
experiment_name: "name"
```
**Of course we can change all of parameters not only one or defult env/model/policy_net etc.**

</details>



<details>
<summary><b>Use CNN nets </b></summary>

Very large number of environments have representation by image. It is then worth using the CNN network, below are some examples of how to do it.
For some experiments you can choose default cnn net from stable-baseline3 library. 
We have to change ```model.policy``` to ```"CnnPolicy"```

This is from file ```config/experiment/pong_with_cnn_ppo.yaml```
```yaml
# @package _global_
defaults:
  - override /env: pong_no_frame_skip
  - override /model: ppo
  - override /policy_net: default

model:
  policy: "CnnPolicy"
  batch_size: 128
  learning_rate: 000.1

experiment_name: "pong_with_cnn_poo"
```
and run from root folder: ```bash bash/run_training.sh```:

```bash
python steps/training.py \
  env=pong_with_cnn_ppo
```

What is more important in many cases, we may want to use some of our own net architecture. 
An example network is ```BasicCustomCNN``` from ```easyRL/custom_policies/policies.py```
You can create your own, just inspire by code of this net.

<details>
<summary><b>Code</b></summary>

You can also go to  ```easyRL/custom_policies/policies.py``` and see.
```python
class BasicCustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 256):
        super(BasicCustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)

        C = observation_space.shape[0]
        H = observation_space.shape[1]
        W = observation_space.shape[2]

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=C, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # take auto number of pixels after cnn
        out_shape = self._get_output_shape_cnn(C, H, W)

        # Define fully connected layers
        self.fc1 = nn.Linear(out_shape, features_dim)

        self._initialize_weights()

    def _get_output_shape_cnn(self, C: int, H: int, W: int) -> int:
        random_observation = torch.rand(1, C, H, W)
        with torch.no_grad():
            x = self.cnn(random_observation)
            x = x.reshape(x.size(0), -1)

        return x.size()[1]

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, gain=init.calculate_gain("relu"))
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight, gain=init.calculate_gain("relu"))
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, observations: Tensor) -> Tensor:
        # Forward pass through the cnn
        x = self.cnn(observations)

        # Flatten the feature map
        x = x.reshape(x.size(0), -1)

        # Go to fc net
        x = torch.relu(self.fc1(x))

        return x
```
</details>

Now, just add this net to ```config/policy_net``` folder.

This is example of file ```custom_cnn.yaml```. If you want to use own net. 
Just change this line:

__target_: easyRL.custom_policies.policies.BasicCustomCNN_

This is how file looks like: 
```yaml
features_extractor_class:
  _target_: easyRL.custom_policies.policies.BasicCustomCNN
  _partial_: true
features_extractor_kwargs:
  features_dim: 128
net_arch: []
```
Note: if you want to read more about custom nets in stable-basleline3
clik here: [link](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html) 
or just trust me that works good 🙈 🙉 🙊

</details>


## How use library for own environments and experiments?
This is probably the most important and interesting part of readme. 
How use easyRL package for your own environment/algorithm/net architecture? 
I hope this is super simply (if not, let me know! 😭) and below you can find several examples:

<details>
<summary><b>How to use easyRL for own envs?</b></summary>

Stable-baseline3 can use only Gymnasium envs, so if you can use this easyRL for all envs from 
[list](https://www.gymlibrary.dev/index.html).

If you want to create custom env this have to written in Gym convention. This shouldn't be 
a big problem, because most of environments are written in this convention.

TODO Duckietown
</details>

<details>
<summary><b>How to add own wrappers?</b></summary>

TODO
</details>


<details>
<summary><b>How to add own RL algorithms like e.g. TRPO? </b></summary>
We use stable-baseline3 library to train models, so we can use only algorithms form this library.
List is quite long, so probably you would find all models you need. 
[Click here to find current list of models](https://stable-baselines3.readthedocs.io/en/master/guide/algos.html)

If you want to add new model, just go to ```config/model/..``` create new yaml file (like other ones) 
and use it 😊 

Unfortunately, not all algorithms are implemented in stable-baseline3 , 
in which case all we can do is cry.... 
or alternatively, use the stable-baseline3 extension package with additional models.

Here you can find link:
https://sb3-contrib.readthedocs.io/en/master/guide/install.html

and list of extra algorithms:
https://sb3-contrib.readthedocs.io/en/master/guide/algos.html

To install just execute:

```
pip install sb3-contrib
```
_Note: It is impossible to not use stable-baseline3 as library with implementation of RL algos_

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

### EasyRL dashboard
RL experiments take a very long time, sometimes even many days/weeks. 
It is good to know how experiment perform and have estimation time when experiments end.
All these features have dashboard created with [streamlit](https://streamlit.io/) package. 
You can run this with simple bash command:

```bash
bash bash/run_dashboard.sh
```
Now, just go to http://localhost:8501 choose your env, algorithm and experiment to see how it performs.

![](https://github.com/karo56/easyRL/blob/images/dashboard_1.png)
When you choose your env/algorithm/experiment you have to click "Refresh experiment" button.
If the experiment is going on you have to click this button to ... refresh experiment 😐 

![](https://github.com/karo56/easyRL/blob/images/dashboard_2.png)

When you refresh experment you can see some nice plots, and estimation when experiments.

_Note: envs, algorithms and experiments are based on outputs/experiments folder. If this folder is empty, you would see nothing special on this dashboard_ 😥 


### Tensorboard
Stable-baseline3 offers use tensorboard to tracking experiments.
EasyRL saves tensorboard logs in ```experiments/env/algo/experiment_name/tensorboard``` folder by default.

So, if you want to use tensorboard just run:

TODO

and go to localhost.

##  F&Q
<details>
<summary><b>Why don't we use any library to track experiments? </b></summary>

As I wrote above, for the sake of simplifying the library, I did not add such a library. Everyone has their own favorite so I encourage you to add according to your needs!
</details>

<details>
<summary><b>Why prepare_env function have default render_mode = "rgb_array" </b></summary>

In older version of Gym library we could choose render model when we made `env.render(mode=xxx)`. 
Now we have to define render mode when env is inited, so we should use "rgb_array" because of `MakeGifCallback`.
(this is callback is written by myself and you can find code here: [callbacks.py](easyRL%2Fcurstom_wrappers%2Fcallbacks.py))
</details>


<details>
<summary><b>How to track experiments? </b></summary>
TODO
</details>

<details>
<summary><b> How to use it package? </b></summary>
TODO
</details>


<details>
<summary><b>TODO </b></summary>
TODO
</details>

[//]: # (TODO: look how beatuy is it, multple env, multple algos, simple code )

## References
If you want to read some good repos, those are my inspirations:
  - https://github.com/krystianfranus/data-science-template
  - https://github.com/ashleve/lightning-hydra-template
