[![lmgym](https://github.com/chai-nexus/lmgym/actions/workflows/lmgym.yaml/badge.svg)](https://github.com/chai-nexus/lmgym/actions/workflows/lmgym.yaml)

# Language Model Gym
This repository records [Chai Research](https://www.chai-research.com)'s
library for training language models which are then deployed to the
[Chai](https://apps.apple.com/us/app/chai-chat-with-ai-bots/id1544750895) app.
Our current framework is based on Reinforcement Learning from Human Feedback,
this repository includes our work on reward modelling and proximal policy
optimization (based on a modified version of trlx). Most of our model training
been augmented with techniques from [DeepSpeed](https://www.deepspeed.ai) as
well as some novel optimizations. We aim to make this repo a centralized and
accessible place to gather techniques for large language models that improves
convertibility and is aligned with user intentions. Some of the dataset
referenced within the repository may not be publicly avaliable.

## Initial Setup
Create a Python 3 virtual environment and activate:
```
virtualenv -p python3 env
source ./env/bin/activate
```
Install requirements by running:
```
pip install -r requirements.txt
```
Then export project to python path:
```
export PYTHONPATH=$PATH_TO_REPO
```
To test the scripts, run `pytest` in the root directory, you may wish to
install `pytest` separately

## Reward Modelling
### Repository setup
Inside the folder `reward_models` you may find the following files and folders:
- `config.py` contains huggingface authentication token and weights and biases
  token, you may wish to either add them inside your environment variable or
  update the file accordingly
- `utils.py` contains some general utility functions
- `experiments` folder contains all the experiments that have been open
  sourced, we recommend training using 4xA40 GPUs
- `custom` folder contains custom callback functions, custom trainers
  instantiated from the transformer Trainer class and some helper functions
  used during training
- `evaluation` folder allows users to configure a best-of-N chatbot which they
  can speak to (by calling the `eval_rewards_api.py` at the root of the
  repository
### Speaking with trained reward models
After completing training, you may wish to manually evaluate your reward model
against existing reward models.
- First, create an experiment under `reward_models/evaluation/experiments`, you
  need to specify a `get_base_model()` (we use GPTJ) and a dictionary mapping
  model name to an reward model object
- Second, run the `eval_rewards_api.py` by invoking `python3 -m IPython -i --
  eval_rewards_api.py talk_to_models --experiment_name $NAME_OF_EXPERIMENT_FILE
  --select_model $ONE_OF_REWARD_MODEL_NAMES`
- This will prompt you to enter model prompt and bot's first message, for each
  user input, you will see N generated responses and their corresponding reward
  model scores / ranks.

