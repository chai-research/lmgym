import logging

from transformers import Trainer, default_data_collator, set_seed
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
import torch

from clm_models.callbacks.callbacks import get_callbacks
from clm_models.custom import losses as custom_loss
from clm_models.custom import training_utils as utils

check_min_version("4.24.0")
require_version("datasets>=1.8.0")

logger = logging.getLogger(__name__)


def main():
    model_args, data_args, train_args, lora_args = utils.get_parsed_arguments()
    utils.setup_logging(train_args.get_process_log_level())
    utils.log_training_environment_info(train_args)

    set_seed(train_args.seed)

    raw_datasets = utils.get_raw_dataset(data_args, model_args)
    raw_datasets = raw_datasets.shuffle(train_args.seed)

    tokenizer = utils.get_tokenizer(model_args)
    model_config = utils.get_model_config(model_args)
    model = utils.get_base_model_for_finetuning(model_args, model_config)
    model.resize_token_embeddings(len(tokenizer))
    if lora_args.use_lora:
        model = utils.prepare_lora_model(model, lora_args)

    tokenized_datasets = utils.get_tokenized_dataset(
        train_args,
        data_args,
        raw_datasets,
        tokenizer
    )
    with train_args.main_process_first(desc="grouping texts together"):
        lm_datasets = tokenized_datasets
        print("Example data:", lm_datasets["train"][0])

    if train_args.do_train:
        train_dataset = utils.get_train_dataset(data_args, lm_datasets)

    if train_args.do_eval:
        eval_dataset = utils.get_eval_dataset(data_args, lm_datasets)

    if data_args.train_to_probs:
        trainer_cls = custom_loss.SoftTargetTrainer
    else:
        trainer_cls = Trainer

    compute_metrics = utils.compute_metrics if train_args.do_eval else None
    logits_preprocessor = utils.preprocess_logits_for_metrics if train_args.do_eval else None
    trainer = trainer_cls(
        model=model,
        args=train_args,
        train_dataset=train_dataset if train_args.do_train else None,
        eval_dataset=eval_dataset if train_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=logits_preprocessor,
        callbacks=get_callbacks(data_args, train_args, tokenizer)
    )
    torch.set_autocast_cache_enabled(False)

    if data_args.eval_first_step:
        # This doesnt work with deepspeed
        print('evaluating first step')
        trainer.evaluate()

    if train_args.do_train:
        checkpoint = None
        last_checkpoint = utils.get_last_checkpoint_if_exists(train_args)
        if train_args.resume_from_checkpoint is not None:
            checkpoint = train_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = utils.get_train_metrics(train_result, data_args, train_dataset)
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if train_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics = utils.get_eval_metrics(metrics, data_args, eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        trainer.log(metrics)

    cards_config = utils.get_huggingface_cards_config(model_args, data_args)
    if train_args.push_to_hub:
        trainer.push_to_hub(**cards_config)
    else:
        trainer.create_model_card(**cards_config)


if __name__ == "__main__":
    main()
