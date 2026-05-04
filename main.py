import os
import time
import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch

import dataloader
import diffusion
import slm
import utils
import wandb
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import CSVLogger


omegaconf.OmegaConf.register_new_resolver("cwd", os.getcwd)
omegaconf.OmegaConf.register_new_resolver("device_count", torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver("eval", eval)
omegaconf.OmegaConf.register_new_resolver("div_up", lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(config, tokenizer):
    if "hf" in config.backbone:
        return diffusion.Diffusion(config, tokenizer=tokenizer).to("cuda")

    if "bfn" in config.backbone or "dit" in config.backbone:
        return slm.Diffusion.load_from_checkpoint(
            config.eval.checkpoint_path,
            tokenizer=tokenizer,
            map_location="cuda",  #
            config=config,
        )

    if "promoter" in config.backbone:
        if config.parameterization == "subs":
            return diffusion.Diffusion.load_from_checkpoint(
                config.eval.checkpoint_path,
                tokenizer=tokenizer,
                map_location="cpu",  #
                config=config,
            )
        else:
            return slm.Diffusion.load_from_checkpoint(
                config.eval.checkpoint_path,
                tokenizer=tokenizer,
                map_location="cpu",  #
                config=config,
            )

    return diffusion.Diffusion.load_from_checkpoint(
        config.eval.checkpoint_path,
        tokenizer=tokenizer,
        map_location="cpu",  #
        config=config,
    )


@L.pytorch.utilities.rank_zero_only
def _init_wandb(config):
    if config.get("wandb", None) is not None:
        config_ = omegaconf.OmegaConf.to_object(config)
        wandb.init(**config_["wandb"])


@L.pytorch.utilities.rank_zero_only
def _print_config(
    config: omegaconf.DictConfig, resolve: bool = True, save_cfg: bool = True
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
      config (DictConfig): Configuration composed by Hydra.
      resolve (bool): Whether to resolve reference fields of DictConfig.
      save_cfg (bool): Whether to save the configuration tree to a file.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    fields = config.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, omegaconf.DictConfig):
            branch_content = omegaconf.OmegaConf.to_yaml(
                config_section, resolve=resolve
            )

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))
    rich.print(tree)
    if save_cfg:
        with fsspec.open(
            "{}/config_tree.txt".format(config.checkpointing.save_dir), "w"
        ) as fp:
            rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, tokenizer, k=500):

    # print train set
    if train_ds is not None:
        print("Printing train dataloader batch.")
        batch = next(iter(train_ds))
        print(
            f"Batch input_ids.shape {batch['input_ids'].shape}, type: {batch['input_ids'].dtype}"
        )
        first = batch["input_ids"][0, :k]
        last = batch["input_ids"][0, -k:]
        last_attn_mask = batch["attention_mask"][0, -k:]
        print(f"First {k} tokens:", tokenizer.decode(first))
        print("ids:", first)
        print(f"Last {k} tokens:", tokenizer.decode(last))
        print(f"Last {k} tokens' attention mask:", last_attn_mask)
        print("ids:", last)

    # print valid set
    if valid_ds is not None:
        print("Printing valid dataloader batch.")
        batch = next(iter(valid_ds))
        print("Batch input_ids.shape", batch["input_ids"].shape)
        first = batch["input_ids"][0, :k]
        last = batch["input_ids"][0, -k:]
        last_attn_mask = batch["attention_mask"][0, -k:]
        print(f"First {k} tokens:", tokenizer.decode(first))
        print("ids:", first)
        print(f"Last {k} tokens:", tokenizer.decode(last))
        print(f"Last {k} tokens' attention mask:", last_attn_mask)
        print("ids:", last)


def write_fasta(output_path, sequences):
    with open(output_path, "w") as f:
        for i, sequence in enumerate(sequences):
            f.write(f">seq#{i} L={len(sequence)}\n{sequence}\n")


def generate_samples(config, logger, tokenizer):
    logger.info("Generating samples.")
    model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
    model.gen_ppl_metric.reset()
    if config.eval.disable_ema:
        logger.info("Disabling EMA.")
        model.ema = None
    stride_length = config.sampling.stride_length
    num_strides = config.sampling.num_strides

    start_time = time.time()
    end_time = None

    for _ in range(config.sampling.num_sample_batches):
        if config.sampling.semi_ar:
            _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
                stride_length=stride_length,
                num_strides=num_strides,
                dt=1 / config.sampling.steps,
            )
            text_samples = intermediate_samples[-1]
            # Note: Samples generated using semi-ar method
            # need to to be processed before computing generative perplexity
            # since these samples contain numerous <|endoftext|> tokens
            # and diffusion.compute_generative_perplexity() discards
            # any text after the first EOS token.
        else:
            samples = model.restore_model_and_sample(num_steps=config.sampling.steps)
            text_samples = model.tokenizer.batch_decode(samples)
            end_time = time.time()
            model.compute_generative_perplexity(text_samples)

    if config.parameterization == "dfm":
        sample_len = len(text_samples[0])
        text_samples = [sample.split("*")[0] for sample in text_samples]
        text_samples = [
            sample for sample in text_samples if (sample != ("!" * len(sample)))
        ]
        text_samples = [
            sample for sample in text_samples if (len(sample) == sample_len)
        ]
        text_samples = [sample for sample in text_samples if ("!" not in sample)]

    print(f"Sampling time: {end_time - start_time:.4f}s")
    print(f"Text samples [{len(text_samples)}] = {text_samples[:5]}")

    filename = f"samples@L={config.sampling.length}_{config.sampling.num_sample_batches * config.loader.eval_batch_size}.fasta"
    write_fasta(os.path.join(config.sampling.outdir, filename), text_samples)

    if not config.sampling.semi_ar:
        print("Generative perplexity:", model.gen_ppl_metric.compute())
        print("Generative KL: ", model.gen_kl_metric.compute())
    return text_samples


def _ppl_eval(config, logger, tokenizer):
    logger.info("Starting Zero Shot Eval.")

    model = _load_from_checkpoint(config=config, tokenizer=tokenizer)
    if config.eval.disable_ema:
        logger.info("Disabling EMA.")
        model.ema = None

    wandb_logger = None
    _init_wandb(config)
    local_logger = CSVLogger(save_dir=config.logging_dir, flush_logs_every_n_steps=100)

    callbacks = []
    if "callbacks" in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))
    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=local_logger,
    )
    _, valid_ds = dataloader.get_dataloaders(
        config, tokenizer, skip_train=True, valid_seed=config.seed
    )
    trainer.validate(model, valid_ds)


def _train(config, logger, tokenizer):
    logger.info("Starting Training.")
    wandb_logger = None
    _init_wandb(config)
    local_logger = CSVLogger(save_dir=config.logging_dir, flush_logs_every_n_steps=100)

    if (
        config.checkpointing.resume_from_ckpt
        and config.checkpointing.resume_ckpt_path is not None
    ):
        ckpt_path = config.checkpointing.resume_ckpt_path
    else:
        ckpt_path = None

    logger.info("here %s %s", ckpt_path, config.checkpointing.resume_ckpt_path)

    # Lightning callbacks
    callbacks = []
    if "callbacks" in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))

    train_ds, valid_ds = dataloader.get_dataloaders(config, tokenizer)
    # _print_batch(train_ds, valid_ds, tokenizer)

    # changed from bfn.Diffusion to diffusion.Diffusion in Rebuttal
    if config.backbone == "FB" or config.backbone == "Mel":
        import slm_enhancer

        model = slm_enhancer.Diffusion(config, tokenizer=train_ds.tokenizer)
    elif config.backbone == "promoter":
        import slm_promoter

        model = slm_promoter.Diffusion(config, tokenizer=train_ds.tokenizer)
    else:
        model = slm.Diffusion(config, tokenizer=train_ds.tokenizer)
    # model = diffusion.Diffusion(config, tokenizer=train_ds.tokenizer)

    if config.backbone == "promoter":
        trainer = hydra.utils.instantiate(
            config.trainer,
            default_root_dir=os.getcwd(),
            callbacks=callbacks,
            strategy=DDPStrategy(find_unused_parameters=True),
            logger=local_logger,
            # use_distributed_sampler=False
        )
    else:
        trainer = hydra.utils.instantiate(
            config.trainer,
            default_root_dir=os.getcwd(),
            callbacks=callbacks,
            strategy=hydra.utils.instantiate(config.strategy),
            logger=local_logger,
            # use_distributed_sampler=False
        )

    trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    """Main entry point for training."""
    L.seed_everything(config.seed)
    _print_config(config, resolve=True, save_cfg=True)

    logger = utils.get_logger(__name__)
    if (
        config.data.train != "promoter"
        and config.data.train != "FB"
        and config.data.train != "Mel"
        and config.data.train != "sudoku"
    ):
        tokenizer = dataloader.get_tokenizer(config)
    else:
        config.callbacks.checkpoint_monitor.monitor = "val/loss"
        tokenizer = None

    if config.mode == "sample_eval":
        generate_samples(config, logger, tokenizer)
    elif config.mode == "ppl_eval":
        _ppl_eval(config, logger, tokenizer)
    else:
        _train(config, logger, tokenizer)


if __name__ == "__main__":
    main()

