import os
import click
import config
from transformers import AutoModelForSequenceClassification


def load_model(model_path):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return model


def get_model_save_path(save_folder, checkpoint):
    folder = f'/models/checkpoints/{save_folder}'
    path = f'{folder}/checkpoint-{checkpoint}'
    return path


def get_hf_save_path(save_folder, checkpoint):
    path = f'ChaiML/{save_folder}_cp_{checkpoint}'
    return path


@click.group(chain=True)
def cli():
    pass


@cli.command(name='upload_model')
@click.option('--save_folder', required=True)
@click.option('--checkpoints', required=True, multiple=True)
def upload_model_to_hf(save_folder, checkpoints):
    for cp in checkpoints:
        path = get_model_save_path(save_folder, cp)
        assert os.path.exists(path)
        model = load_model(path)
        model.eval()
        hf_path = get_hf_save_path(save_folder, cp)
        model.push_to_hub(hf_path, use_auth_token=config.HF_AUTH_TOKEN)
        print(hf_path)


if __name__ == '__main__':
    cli()
