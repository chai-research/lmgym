import re
import logging
import click
import importlib
from datasets import load_dataset

from reward_models.evaluation.evaluator import RewardModelComparitor


DEBUG = True
logger = logging.getLogger(__name__)


@click.group(chain=True)
def cli():
    pass


@cli.command(name='talk_to_models')
@click.option('--experiment_name', required=True)
@click.option('--select_model', required=True)
@click.option('--best_of', required=False, default=6)
@click.option('--ranked', required=False, default=False, is_flag=True)
def talk_to_models(experiment_name, select_model, best_of, ranked):
    experiment = _import_experiment(experiment_name)
    base_model = experiment.get_base_model()
    reward_models = experiment.get_reward_models()
    comparitor = RewardModelComparitor(base_model, reward_models)
    prompt = input('Enter model prompt: ')
    first_message = input('Enter bot first message: ')
    prompt = prompt+f'\nBot: {first_message}'
    bot = ChatBot(prompt, comparitor, best_of, select_model, ranked)
    while True:
        message = input('User: ')
        response = bot.respond(message)
        print(f'Bot: {response}')


@cli.command(name='convo_evolution')
@click.option('--experiment_name', required=True)
def convo_evolution(experiment_name):
    df = _get_eval_convo_data()
    experiment = _import_experiment(experiment_name)
    reward_models = experiment.get_reward_models()
    convo_ids = list(set(df['convo_id']))
    for convo in convo_ids:
        ix = df['convo_id'] == convo
        sub_df = df[ix]
        _print_single_convo_evolution_score(sub_df, reward_models)


def _print_single_convo_evolution_score(sub_df, reward_models):
    for _, row in sub_df.iterrows():
        scores = _get_reward_model_scores(row['inputs'], reward_models)
        _pprint_reward_scores(scores)
        action = input('action (n for next convo): ')
        if action == 'n':
            break


def _get_reward_model_scores(input_text, models):
    out = []
    header, user_resp = _extract_conversation(input_text)
    _pprint_model_input(header, user_resp)
    for name, model in models.items():
        scores = model.predict(header, [user_resp+'\nBot:', '\nBot:'])
        actual_score = scores[0]['score']
        baseline_score = scores[1]['score']
        delta = actual_score - baseline_score
        data = {'name': name, 'delta': delta}
        out.append(data)
    return out


def _pprint_model_input(header, user_resp):
    print('\033[93m'+'*' * 10 + 'BASELINE INPUT' + '*'*10)
    print(header+'\nBot:')
    print('\n'+'*' * 10 + 'MODEL INPUT' + '*'*10)
    print(header+user_resp+'\nBot:\033[0m')


def _extract_conversation(text):
    header = '\nBot:'.join(text.split('\nBot:')[:-1])
    last_user_index = max([m.end() for m in re.finditer("User:", text)])
    return header[:last_user_index], header[last_user_index:]


def _pprint_reward_scores(scores):
    for score in scores:
        color = '\033[92m' if score['delta'] > 0 else '\033[91m'
        print(f'{color}{score}\033[0m')


def _import_experiment(experiment_name):
    module = f'ukml.reward_models.evaluation.experiments.{experiment_name}'
    config = importlib.import_module(module)
    return config


def _get_eval_convo_data():
    path = 'ChaiML/sampled_convos'
    ds = load_dataset(path)['train']
    return ds.to_pandas()


class ChatBot():
    def __init__(self, prompt, model, best_of, select_model, ranked):
        self.chat_history = [prompt]
        self.model = model
        self.best_of = best_of
        self.ranked = ranked
        self.select_model = select_model

    def respond(self, inputs):
        self.chat_history.append(f'User: {inputs}')
        model_input = '\n'.join(self.chat_history)+'\nBot:'
        self._print_debug(model_input)
        out = self.model.predict(model_input, self.best_of)
        df = self.model.pprint(out)
        response = df[self.select_model].idxmax()
        self.chat_history.append(f'Bot: {response.strip()}')
        return response.strip()

    def _print_debug(self, model_input):
        if DEBUG:
            print('#' * 10)
            print(model_input)
            print('#' * 10)


if __name__ == '__main__':
    cli()
