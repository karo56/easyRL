# Output folder
Here you can find output folder. When you run every experiment, all outputs from the experiment including:
- model saved every n steps (frequency of saving you can change in config file, see: ```config/config.yaml```)
- final model
- game gifs files to see how current policy looks like on validation environment (also saved every n steps)
- logger which tracking lengths and rewards of every episode
- params from experiment (in .yaml file)
- tensorboard logger


Every experiment is saved in convention: ```env_name/rl_algo_name/ID_experiment_name```

An example folder looks like this:
![](https://github.com/karo56/easyRL/blob/images/ouptput_folder.png)


_Note: experiment folder is added to .gitignore, so you won't add this heavy files into git tracking_