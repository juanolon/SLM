import itertools
import math
import os
import typing
from dataclasses import dataclass
from slm_probability import CategoricalFactory
from slm_utils import DiscreteBayesianFlow, DiscreteBayesianFlowLoss

from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper
from flow_matching.loss import MixturePathGeneralizedKL

from tqdm import tqdm
import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
import transformers
from torch import Tensor

import dataloader
import noise_schedule
import utils
import wandb
from torchmetrics import KLDivergence


LOG2 = math.log(2)


def _sample_categorical(categorical_probs):
    gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _sample_bernoulli(categorical_probs):
    """
    Sample independent Bernoulli samples for each probability in the input tensor.

    Args:
    - categorical_probs (Tensor): A tensor of shape [bsz, seqlen, vocab] representing probabilities.

    Returns:
    - Tensor: A binary tensor of shape [bsz, seqlen, vocab] with Bernoulli samples.
    """
    # Ensure the input probabilities are of the correct type

    # Sample from uniform distribution of the same shape as categorical_probs
    random_uniform_sample = torch.rand_like(categorical_probs, dtype=torch.float64)

    # Compare random samples to the probabilities to generate Bernoulli samples
    bernoulli_samples = random_uniform_sample < categorical_probs

    return bernoulli_samples


def _unsqueeze(x, reference):
    return x.view(*x.shape, *((1,) * (len(reference.shape) - len(x.shape))))


@dataclass
class Loss:
    loss: torch.FloatTensor
    nlls: torch.FloatTensor
    token_mask: torch.FloatTensor
    reconstruct: torch.FloatTensor  # specified for logging reconstruct loss.
    rnlls: torch.FloatTensor


class NLL(torchmetrics.aggregation.MeanMetric):
    pass


class BPD(NLL):
    def compute(self) -> Tensor:
        """Computes the bits per dimension.

        Returns:
          bpd
        """
        return self.mean_value / self.weight / LOG2


class Perplexity(NLL):
    def compute(self) -> Tensor:
        """Computes the Perplexity.

        Returns:
         Perplexity
        """
        return torch.exp(self.mean_value / self.weight)


class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return torch.softmax(self.model(x, t), dim=-1)


class Diffusion(L.LightningModule):
    def __init__(self, config, tokenizer: transformers.PreTrainedTokenizer):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.tokenizer = tokenizer
        self.vocab_size = (
            self.tokenizer.vocab_size if (self.tokenizer is not None) else 4
        )
        # print("vocab_size is", self.vocab_size)

        self.sampler = self.config.sampling.predictor
        self.gen_ppl_eval_model_name_or_path = (
            self.config.eval.gen_ppl_eval_model_name_or_path
        )
        self.antithetic_sampling = self.config.training.antithetic_sampling
        self.importance_sampling = self.config.training.importance_sampling
        self.change_of_variables = self.config.training.change_of_variables
        self.reconstruct_type = self.config.training.reconstruct_type

        # TODO: we do not need mask token here.
        if (
            not hasattr(self.tokenizer, "mask_token")
            or self.tokenizer.mask_token is None
        ):
            self.mask_index = self.vocab_size
            self.vocab_size += 1
        else:
            self.mask_index = self.tokenizer.mask_token_id
        self.distribution_factory = CategoricalFactory()
        self.bayesian_flow = DiscreteBayesianFlow(
            n_classes=self.vocab_size,
            max_sqrt_beta=self.config.training.beta_bfn,
            onehot_sparse=self.config.training.onehot_sparse,
            top_nums=self.config.training.top_nums,
            linear_trans=self.config.training.linear_trans,
            scheduler=self.config.training.scheduler,
            uniform_encoded=self.config.training.uniform_encoded,
            entropy_to_count=self.config.training.entropy_to_count,
        )  # TODO integrate this to config files
        self.bfn_loss = DiscreteBayesianFlowLoss(
            bayesian_flow=self.bayesian_flow,
            distribution_factory=self.distribution_factory,
        )

        self.fm_scheduler = PolynomialConvexScheduler(n=2.0)
        self.fm_path = MixtureDiscreteProbPath(scheduler=self.fm_scheduler)

        self.fm_loss_fn = MixturePathGeneralizedKL(path=self.fm_path)

        self.parameterization = self.config.parameterization
        if self.config.backbone == "dit":
            from models import dit

            self.backbone = dit.DIT(self.config, vocab_size=self.vocab_size)
        elif self.config.backbone == "dimamba":
            from models import dimamba

            self.backbone = dimamba.DiMamba(
                self.config,
                vocab_size=self.vocab_size,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        elif self.config.backbone == "ar":
            from models import autoregressive

            self.backbone = autoregressive.AR(
                self.config, vocab_size=self.vocab_size, mask_index=self.mask_index
            )
        elif self.config.backbone == "hf_dit":
            self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
                config.eval.checkpoint_path, trust_remote_code=True
            )
        elif (
            self.config.backbone == "dit_bfn"
        ):  # we need a backbone with input adapter now.
            from models import dit_bfn

            self.backbone = dit_bfn.BFN_DIT(self.config, vocab_size=self.vocab_size)
            if self.config.init_pre_embedding:
                from transformers import (
                    BertModel,
                    GPT2Model,
                )

                if "lm1b" in self.config.data.train:
                    model_name = "bert-base-uncased"
                    _pretrained_init = BertModel.from_pretrained(model_name)
                    pretrained_embedded = (
                        _pretrained_init.embeddings.word_embeddings.weight
                    )
                    del _pretrained_init
                    self.backbone.prob_adapter.weight = torch.nn.Parameter(
                        pretrained_embedded.T
                    )
                    print("init with pretrained embedding")
                    if self.config.fixed_embedding:
                        print("embedding has fixed.")
                        for param in self.backbone.prob_adapter.parameters():
                            param.requires_grad = False
                elif "openwebtext" in self.config.data.train:
                    model_name = "GPT2"
                    gpt_model = GPT2Model.from_pretrained(model_name)
                    # tokenizer = GPT2Tokenizer.from_pretrained(model_name)
                    # Step 3: Extract the embedding weights from the GPT model
                    _gpt_embeddings = gpt_model.get_input_embeddings().weight
                    pretrained_embedded = _gpt_embeddings
                    init_tensor = torch.zeros_like(self.backbone.prob_adapter.weight)
                    init_tensor[:, :-1] = pretrained_embedded.T
                    del gpt_model
                    self.backbone.prob_adapter.weight = torch.nn.Parameter(init_tensor)
                    print("init with pretrained embedding")
                    if self.config.fixed_embedding:
                        print("embedding has fixed.")
                        for param in self.backbone.prob_adapter.parameters():
                            param.requires_grad = False
                else:
                    raise NotImplementedError
        elif self.config.backbone == "promoter":
            from models import promoter_model

            self.backbone = promoter_model.PromoterModel(self.config)
        else:
            raise ValueError(f"Unknown backbone: {self.config.backbone}")

        self.T = self.config.T
        self.subs_masking = self.config.subs_masking

        if self.config.parameterization == "dfm":
            self.wrapped_fm_model = WrappedModel(self.backbone)
            self.fm_solver = MixtureDiscreteEulerSolver(
                model=self.wrapped_fm_model,
                path=self.fm_path,
                vocabulary_size=self.vocab_size,
            )

        self.softplus = torch.nn.Softplus()
        # metrics are automatically reset at end of epoch
        metrics = torchmetrics.MetricCollection(
            {"nll": NLL(), "bpd": BPD(), "ppl": Perplexity(), "kl": KLDivergence()}
        )
        metrics.set_dtype(torch.float64)
        self.train_metrics = metrics.clone(prefix="train/")
        self.valid_metrics = metrics.clone(prefix="val/")
        self.valid_recmetrics = metrics.clone(prefix="val_rec/")
        self.test_metrics = metrics.clone(prefix="test/")
        self.prefix = ""

        # generative perplexity
        self.gen_ppl_metric = Perplexity()
        self.gen_kl_metric = KLDivergence()
        self.eval_model_tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.gen_ppl_eval_model_name_or_path
        )
        if self.eval_model_tokenizer.pad_token is None:
            self.eval_model_tokenizer.pad_token = self.eval_model_tokenizer.eos_token
            self.eval_model_tokenizer.pad_token_id = (
                self.eval_model_tokenizer.eos_token_id
            )

        self.noise = noise_schedule.get_noise(self.config, dtype=self.dtype)
        if self.config.training.ema > 0:
            from models import ema

            self.ema = ema.ExponentialMovingAverage(
                itertools.chain(self.backbone.parameters(), self.noise.parameters()),
                decay=self.config.training.ema,
            )
        else:
            self.ema = None

        self.lr = self.config.optim.lr
        self.sampling_eps = self.config.training.sampling_eps
        self.time_conditioning = self.config.time_conditioning
        self.log_interval = self.config.trainer.log_every_n_steps
        self.neg_infinity = -1000000.0
        self.fast_forward_epochs = None
        self.fast_forward_batches = None
        self._validate_configuration()

        # debug
        for name, param in self.named_parameters():
            if not param.requires_grad:
                print(f"Non-trainable param: {name}, Shape: {param.shape}")

    def _validate_configuration(self):
        assert not (self.change_of_variables and self.importance_sampling)
        if self.parameterization == "sedd":
            assert not self.importance_sampling
            assert not self.change_of_variables
        if self.parameterization == "d3pm":
            assert self.T > 0
        if self.T > 0:
            assert self.parameterization in {"d3pm", "subs", "bfn", "new_diff", "dfm"}
        if self.subs_masking:
            assert self.parameterization == "d3pm"

    def on_load_checkpoint(self, checkpoint):
        if self.ema:
            self.ema.load_state_dict(checkpoint["ema"])
        # Copied from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
        self.fast_forward_epochs = checkpoint["loops"]["fit_loop"]["epoch_progress"][
            "current"
        ]["completed"]
        self.fast_forward_batches = checkpoint["loops"]["fit_loop"][
            "epoch_loop.batch_progress"
        ]["current"]["completed"]

    def on_save_checkpoint(self, checkpoint):
        if self.ema:
            checkpoint["ema"] = self.ema.state_dict()
        # Copied from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
        # ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
        # behind, so we're using the optimizer's progress.
        checkpoint["loops"]["fit_loop"]["epoch_loop.batch_progress"]["total"][
            "completed"
        ] = (
            checkpoint["loops"]["fit_loop"][
                "epoch_loop.automatic_optimization.optim_progress"
            ]["optimizer"]["step"]["total"]["completed"]
            * self.trainer.accumulate_grad_batches
        )
        checkpoint["loops"]["fit_loop"]["epoch_loop.batch_progress"]["current"][
            "completed"
        ] = (
            checkpoint["loops"]["fit_loop"][
                "epoch_loop.automatic_optimization.optim_progress"
            ]["optimizer"]["step"]["current"]["completed"]
            * self.trainer.accumulate_grad_batches
        )
        # _batches_that_stepped tracks the number of global steps, not the number
        # of local steps, so we don't multiply with self.trainer.accumulate_grad_batches here.
        checkpoint["loops"]["fit_loop"]["epoch_loop.state_dict"][
            "_batches_that_stepped"
        ] = checkpoint["loops"]["fit_loop"][
            "epoch_loop.automatic_optimization.optim_progress"
        ]["optimizer"]["step"]["total"]["completed"]
        if "sampler" not in checkpoint.keys():
            checkpoint["sampler"] = {}
        if hasattr(self.trainer.train_dataloader.sampler, "state_dict"):
            sampler_state_dict = self.trainer.train_dataloader.sampler.state_dict()
            checkpoint["sampler"]["random_state"] = sampler_state_dict.get(
                "random_state", None
            )
        else:
            checkpoint["sampler"]["random_state"] = None

    def on_train_start(self):
        if self.ema:
            self.ema.move_shadow_params_to_device(self.device)
        # Adapted from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
        distributed = (
            self.trainer._accelerator_connector.use_distributed_sampler
            and self.trainer._accelerator_connector.is_distributed
        )
        if distributed:
            sampler_cls = dataloader.FaultTolerantDistributedSampler
        else:
            sampler_cls = dataloader.RandomFaultTolerantSampler
        updated_dls = []
        for dl in self.trainer.fit_loop._combined_loader.flattened:
            if hasattr(dl.sampler, "shuffle"):
                dl_sampler = sampler_cls(dl.dataset, shuffle=dl.sampler.shuffle)
            else:
                dl_sampler = sampler_cls(dl.dataset)
            if (
                distributed
                and self.fast_forward_epochs is not None
                and self.fast_forward_batches is not None
            ):
                dl_sampler.load_state_dict(
                    {
                        "epoch": self.fast_forward_epochs,
                        "counter": (
                            self.fast_forward_batches * self.config.loader.batch_size
                        ),
                    }
                )
            updated_dls.append(
                torch.utils.data.DataLoader(
                    dl.dataset,
                    batch_size=self.config.loader.batch_size,
                    num_workers=self.config.loader.num_workers,
                    pin_memory=self.config.loader.pin_memory,
                    sampler=dl_sampler,
                    shuffle=False,
                    persistent_workers=False,
                )
            )
        self.trainer.fit_loop._combined_loader.flattened = updated_dls

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema:
            self.ema.update(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )

    def _subs_parameterization(self, logits, xt):
        # log prob at the mask index = - infinity
        logits[:, :, self.mask_index] += self.neg_infinity

        # Normalize the logits such that x.exp() is
        # a probability distribution over vocab_size.
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

        # Apply updates directly in the logits matrix.
        # For the logits of the unmasked tokens, set all values
        # to -infinity except for the indices corresponding to
        # the unmasked tokens.
        unmasked_indices = xt != self.mask_index
        logits[unmasked_indices] = self.neg_infinity
        logits[unmasked_indices, xt[unmasked_indices]] = 0
        return logits

    def _d3pm_parameterization(self, logits):
        if self.subs_masking:
            logits[:, :, self.mask_index] += self.neg_infinity
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        return logits

    def _new_diff_parameterization(self, logits, xt, stage="training"):
        # xt = 0.0
        if stage == "inference":  # only use for inference
            mask = xt > 0
            # mask = mask.to(logits.device)
            logits[~mask] += self.neg_infinity  # mask all outside position

        # logits = torch.softmax(logits,dim=-1)
        if self.config.data.train == "text8":
            # print("softmax here")
            # logits = torch.softmax(logits,dim=-1) #if data is text8, we do not use the logsumexp
            logits = F.log_softmax(logits, dim=-1)
        else:
            logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        return logits
        # logits = torch.where(mask,logits,)

    def _sedd_parameterization(self, logits, xt, sigma):
        esigm1_log = (
            torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1)
            .log()
            .to(logits.dtype)
        )
        # logits shape
        # (batch_size, diffusion_model_input_length, vocab_size)
        logits = logits - esigm1_log[:, None, None] - np.log(logits.shape[-1] - 1)
        # The below scatter operation sets the log score
        # for the input word to 0.
        logits = torch.scatter(
            logits, -1, xt[..., None], torch.zeros_like(logits[..., :1])
        )
        return logits

    def _process_sigma(self, sigma):
        if sigma is None:
            assert self.parameterization == "ar"
            return sigma
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)
        assert sigma.ndim == 1, sigma.shape
        return sigma

    def forward(self, x, sigma, stage="training", signal=None):
        """Returns log score."""
        sigma = self._process_sigma(sigma)

        #   #do sparse operation to the inputs distribution.
        #   keep_numbers = (1 - sigma) *

        with torch.cuda.amp.autocast(dtype=torch.float32):
            if self.config.backbone == "promoter":
                logits = self.backbone(x, sigma, signal=signal)
            else:
                logits = self.backbone(x, sigma)

        if self.parameterization == "new_diff":
            logits = self._new_diff_parameterization(
                logits, x, stage=stage
            )  # masked parameterization.
        # check whether sigma is t

        return logits

    def _d3pm_loss(self, model_output, xt, x0, t):
        dt = 1 / self.T

        if torch.is_tensor(t):
            t = t[:, None]
            assert t.ndim == 2
            t = t.clamp(0.0, 1.0 - 1e-4)
        alpha_t = 1 - t + torch.zeros_like(xt)
        alpha_s = 1 - (t - dt) + torch.zeros_like(xt)

        log_x_theta_at_x0 = torch.gather(model_output, -1, x0[:, :, None]).squeeze(-1)
        log_x_theta_at_m = model_output[:, :, self.mask_index]
        x_theta_at_m = log_x_theta_at_m.exp()

        term_1_coef = dt / t
        term_1_log_nr = torch.log(alpha_t * x_theta_at_m / t + 1)
        term_1_log_dr = log_x_theta_at_x0

        term_2_coef = 1 - dt / t
        term_2_log_nr = term_1_log_nr
        term_2_log_dr = torch.log(alpha_s * x_theta_at_m / (t - dt) + 1)

        L_vb_masked = term_1_coef * (term_1_log_nr - term_1_log_dr) + term_2_coef * (
            term_2_log_nr - term_2_log_dr
        )

        L_vb = L_vb_masked * (xt == self.mask_index)

        return self.T * L_vb

    def _compute_loss(self, batch, prefix):
        if self.config.backbone == "promoter":
            attention_mask = None
            if prefix == "train":
                losses = self._dna_loss(batch, attention_mask)
            elif prefix == "val" or prefix == "test":
                losses = self._dna_valid_loss(batch, attention_mask)
            else:
                raise ValueError(f"Invalid prefix: {prefix}")

            loss = losses.loss
            if prefix == "train":
                self.log(
                    "train/loss",
                    loss.mean().item(),
                    on_step=True,
                    on_epoch=False,
                    sync_dist=True,
                )
            rec_loss = 0
            return loss, rec_loss

        else:
            if "attention_mask" in batch:
                attention_mask = batch["attention_mask"]
            else:
                attention_mask = None

            if prefix == "train":
                losses = self._loss(batch["input_ids"], attention_mask)
            elif prefix == "val" or prefix == "test":
                losses = self._valid_loss(batch["input_ids"], attention_mask)
            else:
                raise ValueError(f"Invalid prefix: {prefix}")

            loss = losses.loss
            if prefix == "train":
                self.train_metrics.update(losses.nlls, losses.token_mask)
            elif prefix == "val":
                self.valid_metrics.update(losses.nlls, losses.token_mask)
                self.valid_recmetrics.update(losses.rnlls, losses.token_mask)
                # metrics = self.valid_metrics
            elif prefix == "test":
                self.test_metrics.update(losses.nlls, losses.token_mask)
                # metrics = self.test_metrics
            else:
                raise ValueError(f"Invalid prefix: {prefix}")
            rec_loss = losses.reconstruct

            return loss, rec_loss

    def on_train_epoch_start(self):
        self.backbone.train()
        self.noise.train()
        self.train_metrics.reset()

    def training_step(self, batch, batch_idx):
        loss, rec = self._compute_loss(batch, prefix="train")

        if self.config.backbone == "promoter":
            return loss

        if self.trainer.global_rank == 0 and self.global_step % self.log_interval == 0:
            wandb.log({"trainer/loss": loss.item()}, self.global_step)
            wandb.log({"trainer/rec": rec.item()}, self.global_step)
        return loss

    def on_train_epoch_end(self):
        current_dict = self.train_metrics.compute()
        self.log_dict(current_dict, on_step=False, on_epoch=True, sync_dist=True)
        if self.trainer.global_rank == 0:
            wandb.log(current_dict, self.current_epoch)

    def on_validation_epoch_start(self):
        if self.ema:
            self.ema.store(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )
            self.ema.copy_to(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )
        self.backbone.eval()
        self.noise.eval()
        self.valid_metrics.reset()
        assert self.valid_metrics.nll.mean_value == 0
        assert self.valid_metrics.nll.weight == 0
        # samples = self._sample()

    def validation_step(self, batch, batch_idx):

        # in protein generation no ppl
        return self._compute_loss(batch, prefix="val")

    def on_validation_epoch_end(self):
        if (
            self.config.eval.compute_perplexity_on_sanity
            or not self.trainer.sanity_checking
        ) and self.config.eval.generate_samples:
            # TODO(justin): implement sampling and kv cache for AR
            samples, text_samples = None, None

            text_samples_collection = []

            for _ in range(self.config.sampling.num_sample_batches):
                samples = self._sample()
                # Decode the samples to be re-tokenized by eval model
                text_samples = self.tokenizer.batch_decode(samples)
                text_samples_collection.extend(text_samples)
                print(
                    f"len of text_samples_collection is {len(text_samples_collection)}"
                )

                if self.config.eval.compute_generative_perplexity:
                    self.compute_generative_perplexity(text_samples)
                    self.compute_generative_kl(text_samples)

            if self.trainer.global_rank == 0:
                # Log the last generated samples
                # text_samples_collection = text_samples_collection[
                #   : self.config.sampling.num_sample_log]

                # modify to pure wandb
                sampled_table = wandb.Table(
                    columns=["Generated Samples"],
                    data=[[s] for s in text_samples_collection],
                )
                wandb.log({f"samples@global_step{self.global_step}": sampled_table})
            ppl_value = self.gen_ppl_metric.compute()
            kl_value = self.gen_kl_metric.compute()
            self.log(
                "val/gen_ppl_epoch",
                ppl_value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "val/gen_kl_epoch",
                kl_value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            if (
                self.config.eval.compute_generative_perplexity
                and self.trainer.global_rank == 0
            ):
                wandb.log({"val/gen_ppl_epoch": ppl_value}, self.global_step)
                wandb.log({"val/gen_kl_epoch": kl_value}, self.global_step)

            # log sudoku
            if self.config.data.train == "sudoku":
                valid_count = 0
                total_violations = 0
                formatted_data = []

                for s in text_samples_collection:
                    is_valid, violations = check_sudoku_validity(s)
                    if is_valid:
                        valid_count += 1
                    total_violations += violations

                    pretty_grid = "\n".join([s[i : i + 9] for i in range(0, 81, 9)])
                    formatted_data.append([pretty_grid, is_valid, violations])

                validity_rate = valid_count / len(text_samples_collection)
                avg_violations = total_violations / len(text_samples_collection)

                wandb.log({"val/sudoku_validity_rate": validity_rate}, self.global_step)
                wandb.log({"val/avg_rule_violations": avg_violations}, self.global_step)

                if self.trainer.global_rank == 0:
                    sampled_table = wandb.Table(
                        columns=["Grid View", "Perfect", "Violations"],
                        data=formatted_data,
                    )
                    wandb.log(
                        {f"sudoku_samples_step_{self.global_step}": sampled_table}
                    )

        valid_dict = self.valid_metrics.compute()
        self.log_dict(valid_dict, on_step=False, on_epoch=True, sync_dist=True)
        valid_rec_dict = self.valid_recmetrics.compute()
        print(f"valid_rec_dict is {valid_rec_dict}")
        self.log_dict(valid_rec_dict, on_step=False, on_epoch=True, sync_dist=True)

        if self.trainer.global_rank == 0:
            wandb.log(valid_dict, self.global_step)
            wandb.log(valid_rec_dict, self.global_step)
        if self.ema:
            self.ema.restore(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )

    def on_test_epoch_start(self):
        self.test_metrics.reset()

    def on_test_epoch_end(self):
        test_dict = self.test_metrics.compute()
        self.log_dict(test_dict, on_step=False, on_epoch=True, sync_dist=True)
        if self.trainer.global_rank == 0:
            wandb.log(test_dict, self.current_epoch)

    def configure_optimizers(self):
        # TODO(yair): Lightning currently giving this warning when using `fp16`:
        #  "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
        #  Not clear if this is a problem or not.
        #  See: https://github.com/Lightning-AI/pytorch-lightning/issues/5558
        optimizer = torch.optim.AdamW(
            itertools.chain(self.backbone.parameters(), self.noise.parameters()),
            lr=self.config.optim.lr,
            betas=(self.config.optim.beta1, self.config.optim.beta2),
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay,
        )

        scheduler = hydra.utils.instantiate(
            self.config.lr_scheduler, optimizer=optimizer
        )
        scheduler_dict = {
            "scheduler": scheduler,
            "interval": "step",
            "monitor": "val/loss",
            "name": "trainer/lr",
        }
        return [optimizer], [scheduler_dict]

    @torch.no_grad()
    def eval_retokenize(self, text_samples, max_length):
        """Retokenizes samples for the eval model.

        Args:
            text_samples: List of sentences generated by the model.
        Returns:
            samples: Samples re-tokenized for the eval model
            attn_mask: Attention mask for the eval model
            eval_context_size: Size of the context for the eval model
        """
        if "llama2" in self.gen_ppl_eval_model_name_or_path:
            tokenizer_kwargs = {
                "text_samples": text_samples,
                "return_tensors": "pt",
                "return_token_type_ids": False,
                "return_attention_mask": True,
                "truncation": True,
                "padding": True,
                "max_length": max_length,
            }
            eval_context_size = 4096
        else:
            tokenizer_kwargs = {
                "return_tensors": "pt",
                "return_token_type_ids": False,
                "return_attention_mask": True,
                "truncation": True,
                "padding": True,
                "max_length": max_length,
            }
            eval_context_size = 1024
        samples = self.eval_model_tokenizer(text_samples, **tokenizer_kwargs)
        attn_mask = samples["attention_mask"]
        samples = samples["input_ids"]
        if "llama2" not in self.gen_ppl_eval_model_name_or_path:
            attn_mask = attn_mask.to(self.device)
            samples = samples.to(self.device)
        return samples, attn_mask, eval_context_size

    @torch.no_grad()
    def compute_generative_perplexity(
        self,
        text_samples: typing.List[str],
        retokenize: bool = True,
        max_length: typing.Optional[int] = None,
    ) -> None:
        """Compute the generative perplexity of the model.

        Args:
            text_samples: List of sentences generated by the model.

        Returns:
            Perplexity of the generated text under a different
            pre-trained AR model (e.g., GPT2).
        """
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        eval_model = transformers.AutoModelForCausalLM.from_pretrained(
            self.gen_ppl_eval_model_name_or_path
        ).eval()
        if max_length is None:
            max_length = self.config.model.length
        if "llama2" not in self.gen_ppl_eval_model_name_or_path:
            eval_model = eval_model.to(self.device)
        # Re-tokenize using eval model's tokenizer
        if retokenize:
            (samples, attn_mask, eval_context_size) = self.eval_retokenize(
                text_samples, max_length=max_length
            )
        else:
            samples = text_samples
            attn_mask = torch.ones(samples.shape).to(self.device)
            eval_context_size = samples.shape[-1]
        batch_size = min(self.config.eval.perplexity_batch_size, samples.shape[0])
        num_batches = samples.shape[0] // batch_size
        for i in range(num_batches):
            _samples = torch.split(
                samples[i * batch_size : (i + 1) * batch_size],
                eval_context_size,
                dim=-1,
            )
            _attn_mask = torch.split(
                attn_mask[i * batch_size : (i + 1) * batch_size],
                eval_context_size,
                dim=-1,
            )
            for sample_chunk, attn_mask_chunk in zip(_samples, _attn_mask):
                logits = eval_model(sample_chunk, attention_mask=attn_mask_chunk)[0]
                logits = logits.transpose(-1, -2)

                nlls = F.cross_entropy(
                    logits[..., :-1], sample_chunk[..., 1:], reduction="none"
                )
                first_eos = (
                    sample_chunk == self.eval_model_tokenizer.eos_token_id
                ).cumsum(-1) == 1
                token_mask = sample_chunk != self.eval_model_tokenizer.eos_token_id
                self.gen_ppl_metric.update(
                    nlls, first_eos[..., 1:] + token_mask[..., 1:]
                )

    def q_xt(self, x, move_chance):
        """Computes the noisy sample xt.

        Args:
          x: int torch.Tensor with shape (batch_size,
              diffusion_model_input_length), input.
          move_chance: float torch.Tensor with shape (batch_size, 1).
        """
        move_indices = torch.rand(*x.shape, device=x.device) < move_chance
        xt = torch.where(move_indices, self.mask_index, x)
        return xt

    def _sample_prior(self, *batch_dims):
        return self.mask_index * torch.ones(*batch_dims, dtype=torch.int64)

    def _ddpm_caching_update(self, x, t, dt, p_x0=None):
        assert self.config.noise.type == "loglinear"
        sigma_t, _ = self.noise(t)
        if t.ndim > 1:
            t = t.squeeze(-1)
        assert t.ndim == 1
        move_chance_t = t[:, None, None]
        move_chance_s = (t - dt)[:, None, None]
        assert move_chance_t.ndim == 3, move_chance_t.shape
        if p_x0 is None:
            p_x0 = self.forward(x, sigma_t).exp()

        assert move_chance_t.ndim == p_x0.ndim
        q_xs = p_x0 * (move_chance_t - move_chance_s)
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        _x = _sample_categorical(q_xs)

        copy_flag = (x != self.mask_index).to(x.dtype)
        return p_x0, copy_flag * x + (1 - copy_flag) * _x

    def _ddpm_update(self, x, t, dt):
        sigma_t, _ = self.noise(t)
        sigma_s, _ = self.noise(t - dt)
        if sigma_t.ndim > 1:
            sigma_t = sigma_t.squeeze(-1)
        if sigma_s.ndim > 1:
            sigma_s = sigma_s.squeeze(-1)
        assert sigma_t.ndim == 1, sigma_t.shape
        assert sigma_s.ndim == 1, sigma_s.shape
        move_chance_t = 1 - torch.exp(-sigma_t)
        move_chance_s = 1 - torch.exp(-sigma_s)
        move_chance_t = move_chance_t[:, None, None]
        move_chance_s = move_chance_s[:, None, None]
        unet_conditioning = sigma_t
        log_p_x0 = self.forward(x, unet_conditioning)
        assert move_chance_t.ndim == log_p_x0.ndim
        # Technically, this isn't q_xs since there's a division
        # term that is missing. This division term doesn't affect
        # the samples.
        q_xs = log_p_x0.exp() * (move_chance_t - move_chance_s)
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        _x = _sample_categorical(q_xs)

        copy_flag = (x != self.mask_index).to(x.dtype)
        return copy_flag * x + (1 - copy_flag) * _x

    def _ar_sampler(self, bsz):
        # precompute token buffer
        num_pred_tokens = self.config.model.length - 1
        x = torch.zeros(
            (bsz, num_pred_tokens + 1), dtype=torch.long, device=self.device
        )
        x[:, 0] = self.tokenizer.bos_token_id
        # precompute noise
        noise = (
            torch.distributions.Gumbel(0, 1)
            .sample((bsz, num_pred_tokens, self.vocab_size))
            .to(self.device)
        )
        for i in range(num_pred_tokens):
            next_logits = self.forward(x[:, : i + 1], None)[:, -1]
            y = (next_logits + noise[:, i]).argmax(-1)
            x[:, i + 1] = y
        return x

    def _bfn_sampler(self, bsz, numsteps):
        n_steps = numsteps
        data_shape = (bsz, self.config.model.length)
        input_params = self.bayesian_flow.get_prior_input_params(
            data_shape, self.device
        )

        distribution_factory = self.distribution_factory
        for i in range(1, n_steps + 1):
            t = torch.ones(bsz, device=self.device) * (i - 1) / n_steps
            output_params = self.forward(
                self.bayesian_flow.params_to_net_inputs(input_params, t), t
            )
            output_sample = distribution_factory.get_dist(
                output_params, input_params, t
            ).sample()
            # output_sample = distribution_factory.get_dist(output_params, input_params, t).mode
            output_sample = output_sample.reshape(*data_shape)
            input_params = self.bayesian_flow(
                output_sample, t.unsqueeze(-1)
            )  # change sampling
            alpha = self.bayesian_flow.get_alpha(i, n_steps)
            y = self.bayesian_flow.get_sender_dist(output_sample, alpha).sample()
            input_params = self.bayesian_flow.update_input_params(
                input_params, y, alpha
            )

        t = torch.ones(bsz, device=self.device)
        output_params = self.forward(
            self.bayesian_flow.params_to_net_inputs(input_params, t), t
        )
        output_sample = distribution_factory.get_dist(
            output_params, input_params, t
        ).mode
        output_sample = output_sample.reshape(*data_shape)
        return output_sample

    def _sample_newdiff(self, bsz, numsteps):

        x_t = (
            torch.ones(bsz, self.config.sampling.length, self.vocab_size)
            / self.vocab_size
        ).to(self.device)  # [N, K] discrete prior
        print(f"sample steps {numsteps}")
        for i in tqdm(range(1, numsteps + 1), desc="sampling"):
            t = torch.ones(bsz, 1).to(x_t.device) * (i - 1) / numsteps
            t = 1 - t  # reverse
            #   # [B,1,1]
            print(f"x_t: {x_t.shape}, t: {t.shape}")
            model_output = self.forward(x_t, t, stage="inference")
            print(f"output t={i}, shape:{model_output.shape}:\n{model_output}")
            model_output = torch.exp(model_output)
            # print("model output",model_output, t)
            t = t.unsqueeze(-1)
            # model_output = torch.log(model_output)
            nominator = torch.clamp(self.expected_nums(t - 1 / numsteps) - 1, min=1e-1)
            denominator = torch.clamp(self.expected_nums(t) - 1, min=1e-1)
            predicted = model_output * 1 + torch.clamp(
                nominator / denominator, max=1.0, min=0.0
            ) * (1 - model_output)
            predicted = torch.clamp(predicted, min=0.0, max=1.0)
            sample_pred = _sample_bernoulli(predicted) * (x_t > 0)
            # sample_pred = torch.distributions.Bernoulli(predicted).sample() * (x_t > 0)
            sample_pred_sum = sample_pred.sum(-1, keepdim=True)
            mask = sample_pred_sum > 0
            sample_pred = torch.where(
                mask,
                sample_pred,
                F.one_hot(predicted.argmax(-1), num_classes=self.vocab_size),
            )
            x_t = sample_pred / torch.clamp(sample_pred.sum(-1, keepdim=True), min=0.0)

        t = (torch.zeros(bsz, 1) + 1 / numsteps).to(x_t.device)
        predicted = self.forward(x_t, t, stage="inference")
        sample_pred = torch.argmax(predicted, dim=-1)
        print(f"final: {sample_pred.shape}, {sample_pred}")

        return sample_pred

    def _sample_dfm(self, bsz, numsteps):

        if self.config.prior == "uniform":
            x_init = torch.randint(
                size=(bsz, self.config.sampling.length),
                high=self.vocab_size,
                device=self.device,
            )
        elif self.config.prior == "masked":
            x_init = (
                torch.zeros(size=(bsz, self.config.sampling.length), device=self.device)
                + self.mask_index
            ).long()

        else:
            raise ValueError(f"Unknown prior: {self.config.prior})")

        linspace_to_plot = torch.linspace(0, 1 - self.config.optim.eps, 9)
        sol_output = self.fm_solver.sample(
            x_init=x_init,
            step_size=(1 / numsteps),
            verbose=True,
            return_intermediates=True,
            time_grid=linspace_to_plot,
        )

        print(f"sol: {sol_output.shape}")

        return sol_output[-1]

    def compute_reconstruction_loss(self, data: Tensor, type="ce") -> Tensor:
        bsz = data.shape[0]
        t = torch.ones(bsz, device=self.device).float().unsqueeze(-1)
        input_params = self.bayesian_flow(data, t)
        net_inputs = self.bayesian_flow.params_to_net_inputs(input_params, t)
        output_params: Tensor = self.forward(net_inputs, t)
        # print("shape here",output_params.shape)
        # if type == "ce":
        #         return self.loss.reconstruction_loss(data, output_params, input_params).flatten(start_dim=1).mean()
        #     elif type == "l2":
        #         return self.loss.cts_time_loss(data, output_params.float(), input_params, t)
        #     else:
        #         raise NotImplementedError
        if type == "ce":
            return self.bfn_loss.reconstruction_loss(
                data, output_params, input_params
            ).flatten(start_dim=1)
        elif type == "l2":
            # output_params = F.one_hot(torch.argmax(output_params,dim=-1),num_classes=self.vocab_size) #check it to onehot.
            # output_params = F.one_hot(torch.argmax(input_params[0],dim=-1),num_classes=self.vocab_size)
            # print((data == output_params.argmax(-1)).float().mean())
            return self.bfn_loss.cts_time_loss(
                data, output_params.float(), input_params, t
            )
        else:
            raise NotImplementedError

    @torch.no_grad()
    def _sample(self, num_steps=None, eps=1e-5):
        """Generate samples from the model."""
        batch_size_per_gpu = self.config.loader.eval_batch_size
        if self.parameterization == "ar":
            return self._ar_sampler(batch_size_per_gpu)
        if num_steps is None:
            num_steps = self.config.sampling.steps
        if self.parameterization == "bfn":
            return self._bfn_sampler(batch_size_per_gpu, num_steps)  # Add BFN sampler
        if self.parameterization == "new_diff":
            return self._sample_newdiff(
                batch_size_per_gpu, num_steps
            )  # add new_diff sampler
        if self.parameterization == "dfm":
            return self._sample_dfm(batch_size_per_gpu, num_steps)
        # Lightning auto-casting is not working in this method for some reason
        x = self._sample_prior(batch_size_per_gpu, self.config.sampling.length).to(
            self.device
        )
        timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
        dt = (1 - eps) / num_steps
        p_x0_cache = None

        for i in range(num_steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)
            if self.sampler == "ddpm":
                x = self._ddpm_update(x, t, dt)
            elif self.sampler == "ddpm_cache":
                p_x0_cache, x_next = self._ddpm_caching_update(
                    x, t, dt, p_x0=p_x0_cache
                )
                if not torch.allclose(x_next, x) or self.time_conditioning:
                    # Disable caching
                    p_x0_cache = None
                x = x_next
            else:
                x = self._analytic_update(x, t, dt)

        if self.config.sampling.noise_removal:
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
            if self.sampler == "analytic":
                x = self._denoiser_update(x, t)
            else:
                unet_conditioning = self.noise(t)[0]
                x = self.forward(x, unet_conditioning).argmax(dim=-1)
        return x

    def restore_model_and_sample(self, num_steps, eps=1e-5):
        """Generate samples from the model."""
        # Lightning auto-casting is not working in this method for some reason
        if self.ema:
            self.ema.store(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )
            self.ema.copy_to(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )
        self.backbone.eval()
        self.noise.eval()
        samples = self._sample(num_steps=num_steps, eps=eps)
        if self.ema:
            self.ema.restore(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )
        self.backbone.train()
        self.noise.train()
        return samples

    def get_score(self, x, sigma):
        model_output = self.forward(x, sigma)
        if self.parameterization == "subs":
            log_k = -torch.log(torch.expm1(sigma)).squeeze(-1)
            assert log_k.ndim == 1

            masked_score = model_output + log_k[:, None, None]
            masked_score[:, :, self.mask_index] = 0

            unmasked_score = self.neg_infinity * torch.ones_like(model_output)
            unmasked_score = torch.scatter(
                unmasked_score,
                -1,
                x[..., None],
                torch.zeros_like(unmasked_score[..., :1]),
            )
            unmasked_score[:, :, self.mask_index] = -(
                log_k[:, None] * torch.ones_like(x)
            )

            masked_indices = (x == self.mask_index).to(model_output.dtype)[:, :, None]
            model_output = masked_score * masked_indices + unmasked_score * (
                1 - masked_indices
            )
        return model_output.exp()

    def _staggered_score(self, score, dsigma):
        score = score.clone()
        extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
        score *= dsigma.exp()[:, None]
        score[..., self.mask_index] += extra_const
        return score

    def _analytic_update(self, x, t, step_size):
        curr_sigma, _ = self.noise(t)
        next_sigma, _ = self.noise(t - step_size)
        dsigma = curr_sigma - next_sigma
        score = self.get_score(x, curr_sigma)
        stag_score = self._staggered_score(score, dsigma)
        probs = stag_score * self._transp_transition(x, dsigma)
        return _sample_categorical(probs)

    def _denoiser_update(self, x, t):
        sigma, _ = self.noise(t)
        score = self.get_score(x, sigma)
        stag_score = self._staggered_score(score, sigma)
        probs = stag_score * self._transp_transition(x, sigma)
        probs[..., self.mask_index] = 0
        samples = _sample_categorical(probs)
        return samples

    def _transp_transition(self, i, sigma):
        sigma = _unsqueeze(sigma, reference=i[..., None])
        edge = torch.exp(-sigma) * F.one_hot(i, num_classes=self.vocab_size)
        edge += torch.where(i == self.mask_index, 1 - torch.exp(-sigma).squeeze(-1), 0)[
            ..., None
        ]
        return edge

    def _sample_t(self, n, device):
        _eps_t = torch.rand(n, device=device)
        if self.antithetic_sampling:
            offset = torch.arange(n, device=device) / n
            _eps_t = (_eps_t / n + offset) % 1
        t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
        if self.importance_sampling:
            return self.noise.importance_sampling_transformation(t)
        return t

    def _maybe_sub_sample(self, x0, attention_mask):
        seqlen = x0.shape[1]
        # print(f"seqlen: {seqlen}")

        if seqlen > self.config.model.length:
            assert seqlen == 2 * self.config.model.length

            # cropping is needed for text8-crop dataset
            # try the same starting point for now
            start = np.random.choice(self.config.model.length)
            end = start + self.config.model.length
            input_tokens = x0[:, start:end]
            output_tokens = x0[:, start + 1 : end + 1]
            new_attention_mask = attention_mask[:, start:end]

            # Helps with validation PPL, since the val
            # examples will all start and end with BOS/EOS
            input_tokens[:, 0] = self.tokenizer.bos_token_id
            output_tokens[:, -1] = self.tokenizer.eos_token_id
        elif self.parameterization == "ar":
            input_tokens = x0[:, :-1]
            output_tokens = x0[:, 1:]
            new_attention_mask = attention_mask[:, 1:]
        else:
            input_tokens = x0
            output_tokens = None
            new_attention_mask = attention_mask
        return input_tokens, output_tokens, new_attention_mask

    def _reconstruction_loss(self, x0):
        t0 = torch.zeros(x0.shape[0], dtype=self.dtype, device=self.device)
        assert self.config.noise.type == "loglinear"
        # The above assert is for d3pm parameterization
        unet_conditioning = self.noise(t0)[0][:, None]
        model_output_t0 = self.forward(x0, unet_conditioning)
        return -torch.gather(
            input=model_output_t0, dim=-1, index=x0[:, :, None]
        ).squeeze(-1)

    def _bfn_forward_loss(self, x0, stage="training"):
        if not self.config.training.different_time:
            t = torch.rand(1, device=x0.device).unsqueeze(-1)
            t = t.repeat(x0.shape[0], 1)
        else:
            t = torch.rand(x0.shape[0], device=x0.device).unsqueeze(-1)
        input_params = self.bayesian_flow(x0, t)
        net_inputs = self.bayesian_flow.params_to_net_inputs(input_params, t)

        output_params: Tensor = self.forward(net_inputs, t)
        if stage == "training":
            loss = self.bfn_loss.cts_time_loss(
                x0, output_params.float(), input_params, t, stage
            )
        elif stage == "inference":
            loss = self.bfn_loss.cts_time_loss(
                x0, output_params.float(), input_params, t, stage
            )
            # t = torch.randint(0, self.T, (x0.size(0),), device=x0.device).unsqueeze(-1) /  self.T
            # with torch.no_grad():
            #   loss = self.bfn_loss.discrete_time_loss(x0,output_params.float(),input_params,t,self.T,n_samples=1)
        else:
            raise NotImplementedError
        # loss shape is (batch_size, 1)
        # print(loss)
        return loss

    def get_xt(self, x0, t):
        x0 = F.one_hot(x0.long(), num_classes=self.vocab_size)
        nums = self.log_linear_fun(t)  # [Batch, 1]
        ones_to_mask = self.generate_3d_tensor_with_ones(
            x0.shape, nums
        )  # [Batch, Seq, Features]
        xt = torch.where(x0 == 1, x0, ones_to_mask)  # [Batch, Seq, Features]

        return xt

    def expected_nums(self, t):
        if self.config.training.bscheduler == "loglinear":
            return torch.clamp(torch.exp(math.log(self.vocab_size) * t), min=1.0)
        elif self.config.training.bscheduler == "linear":
            return torch.clamp(self.vocab_size * t, min=1.0)
        else:
            raise NotImplementedError

    def get_xt_bernoulli(self, x0, t):
        expect_nums = self.expected_nums(t)
        x0 = F.one_hot(x0.long(), num_classes=self.vocab_size)
        bernoulli_param = (expect_nums - 1) / self.vocab_size
        bernoulli_param = bernoulli_param.unsqueeze(-1).repeat(
            1, x0.shape[1], x0.shape[2]
        )  # Batch, seq, Features
        samples = torch.distributions.Bernoulli(probs=bernoulli_param).sample()
        xt = torch.where(x0 == 1, x0, samples)  # [Batch, Seq, Features]
        xt = xt / xt.sum(-1, keepdim=True)  # normalize this

        return xt

    def _forward_new_diffusion(self, x0, stage="training"):
        t = self._sample_t(x0.shape[0], x0.device).unsqueeze(-1)  # [Batch, 1]
        if stage == "debugging":  # here we tends to demonstrate the
            for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                t = torch.ones_like(t) * i
                x_t = self.get_xt_bernoulli(x0, t)  # [Batch, Seq, Features]
                model_output = self.forward(x_t, t, stage="inference")
                output_entropy = -(
                    (torch.exp(model_output) * model_output) * (x_t > 0)
                ).sum(-1)
                normalized_input = x_t / x_t.sum(-1, keepdim=True) + 1e-6
                input_entropy = -(
                    normalized_input * torch.log(normalized_input) * (x_t > 0)
                ).sum(-1)
                diffusion_loss = self._new_diffusion_loss(
                    model_output=model_output, xt=x_t, x0=x0, t=t, stage="inference"
                )
                # x_t = x_t / x_t.sum(-1,keepdim=True)
                if self.trainer.global_rank == 0:
                    print(
                        "t:",
                        t.mean().item(),
                        "diffusion_loss:",
                        diffusion_loss.mean().item(),
                        "o_entropy:",
                        output_entropy.mean().item(),
                        "i_entropy",
                        input_entropy.mean().item(),
                    )
            return diffusion_loss

        if self.T > 0:
            t = (t * self.T).to(torch.int)
            t = t / self.T
            # t \in {1/T, 2/T, ..., 1}
            t += 1 / self.T

        x_t = self.get_xt_bernoulli(x0, t)  # [Batch, Seq, Features]
        # x_t = x_t / x_t.sum(-1,keepdim=True)
        # t = t.unsqueeze(-1)  # [B,1,1]
        # t = t.repeat(1, x_t.size(1), 1)
        model_output = self.forward(x_t, t, stage=stage)
        # utils.print_nans(model_output, 'model_output')
        if self.T > 0:
            diffusion_loss = self._new_diffusion_loss(
                model_output=model_output, xt=x_t, x0=x0, t=t, stage=stage
            )
        return diffusion_loss

    def _dna_forward_new_diffusion(self, x0, stage="training"):
        print("calling dna forward for new diffusion")

        t = self._sample_t(x0.shape[0], x0.device).unsqueeze(-1)  # [Batch, 1]
        if stage == "debugging":  # here we tends to demonstrate the
            for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                t = torch.ones_like(t) * i
                x_t = self.get_xt_bernoulli(x0, t)  # [Batch, Seq, Features]
                model_output = self.forward(x_t, t, stage="inference")
                output_entropy = -(
                    (torch.exp(model_output) * model_output) * (x_t > 0)
                ).sum(-1)
                normalized_input = x_t / x_t.sum(-1, keepdim=True) + 1e-6
                input_entropy = -(
                    normalized_input * torch.log(normalized_input) * (x_t > 0)
                ).sum(-1)
                diffusion_loss = self._new_diffusion_loss(
                    model_output=model_output, xt=x_t, x0=x0, t=t, stage="inference"
                )
                # x_t = x_t / x_t.sum(-1,keepdim=True)
                if self.trainer.global_rank == 0:
                    print(
                        "t:",
                        t.mean().item(),
                        "diffusion_loss:",
                        diffusion_loss.mean().item(),
                        "o_entropy:",
                        output_entropy.mean().item(),
                        "i_entropy",
                        input_entropy.mean().item(),
                    )
            return diffusion_loss

        if self.T > 0:
            t = (t * self.T).to(torch.int)
            t = t / self.T
            # t \in {1/T, 2/T, ..., 1}
            t += 1 / self.T
        signal = x0[:, :, 4:5]
        x0 = x0[:, :, :4]
        x0 = torch.argmax(x0, dim=-1)
        x_t = self.get_xt_bernoulli(x0, t)  # [Batch, Seq, Features]
        # x_t = x_t / x_t.sum(-1,keepdim=True)
        # t = t.unsqueeze(-1)  # [B,1,1]
        # t = t.repeat(1, x_t.size(1), 1)
        model_output = self.forward(x_t, t, stage=stage, signal=signal)
        # utils.print_nans(model_output, 'model_output')
        if self.T > 0:
            diffusion_loss = self._new_diffusion_loss(
                model_output=model_output, xt=x_t, x0=x0, t=t, stage=stage
            )
        else:
            diffusion_loss = 0
        return diffusion_loss

    def _forward_dfm(self, x0, stage="training"):
        t = torch.rand(x0.shape[0]).to(x0.device) * (1 - self.config.optim.eps)

        if self.config.prior == "uniform":
            x1 = torch.randint_like(x0, high=self.vocab_size)
        elif self.config.prior == "masked":
            # print("masked")
            x1 = torch.ones_like(x0) * self.mask_index
        else:
            raise NotImplementedError

        path_sample = self.fm_path.sample(t=t, x_0=x1, x_1=x0)

        model_output = self.forward(path_sample.x_t, path_sample.t, stage=stage)

        fm_loss = self.fm_loss_fn(
            logits=model_output, x_1=x0, x_t=path_sample.x_t, t=path_sample.t
        )

        return fm_loss

    def _new_diffusion_loss(self, model_output, xt, x0, t, stage="training"):
        dt = 1 / self.T
        #   n_t = self.log_linear_fun(t).unsqueeze(-1) #make this part continous
        #   n_t_1 = self.log_linear_fun(t-dt).unsqueeze(-1)
        mask = xt > 0

        n_t = self.expected_nums(t).unsqueeze(-1)

        n_t_1 = self.expected_nums(t - 1 / self.T).unsqueeze(-1)

        nominator = torch.clamp(n_t_1 - 1, min=1e-1)
        denominator = torch.clamp(n_t - 1, min=1e-1)

        one_hot_x0 = F.one_hot(x0, num_classes=self.vocab_size)
        if self.config.data.train == "text8":
            # for text8 we do not use the logsumexp:
            # model_output = torch.log(model_output)
            model_output = model_output

        predicted = torch.exp(model_output) * 1 + torch.clamp(
            nominator / denominator, min=0.0, max=1.0 - 1e-4
        ) * (1 - torch.exp(model_output))  # model_output is probability

        if stage == "training":
            nlog_p = -torch.gather(model_output, -1, x0[:, :, None]).squeeze(-1)
            if self.config.training.without_T:
                nlog_p = nlog_p
            else:
                nlog_p = self.T * nlog_p
        elif stage == "inference":
            # if self.config.c
            if self.config.eval.new_diff_calculate == "argmax":
                max_index = torch.argmax(model_output, dim=-1)
                p_max = torch.gather(predicted, -1, max_index[:, :, None])
                nlog_max = -torch.gather(
                    torch.log(predicted), -1, max_index[:, :, None]
                )
                weight = torch.clamp(nominator / denominator, min=0.0, max=1.0 - 1e-6)
                p_max = torch.clamp(p_max, max=1 - 1e-6)
                bernoulli_kl = (
                    (weight) * torch.log(weight) + (1 - weight) * torch.log(1 - weight)
                ) - ((weight) * torch.log(p_max) + (1 - weight) * torch.log(1 - p_max))
                cross_entropy_false = -torch.log(weight) + bernoulli_kl
                cross_entropy_true = -(
                    torch.gather(torch.log(predicted), -1, x0[:, :, None])
                )
                true_index = (torch.argmax(model_output, dim=-1) == x0).unsqueeze(-1)
                # print("true_index",true_index.float().squeeze(-1).mean(-1))
                # print("cross_entropy_true",cross_entropy_true.float().squeeze(-1).mean(-1))
                # print("cross_entropy_false",cross_entropy_false.float().squeeze(-1).mean(-1))
                # print(torch.exp(model_output).max(-1))
                # correct_index =
                cross_entropy_true = cross_entropy_true * true_index
                cross_entropy_false = cross_entropy_false * (~true_index)
                nlog_p = (cross_entropy_true + cross_entropy_false).squeeze(-1)
                nlog_p = nlog_p * self.T
            elif self.config.eval.new_diff_calculate == "full":
                mask = xt > 0
                predicted = torch.clamp(predicted, min=1e-6, max=1.0 - 1e-6)
                weight = torch.clamp(nominator / denominator, min=0.0, max=1.0)
                onehot = F.one_hot(x0, num_classes=self.vocab_size)
                # weight = weight.repeat(1,predicted.size(1),predicted.size(2))
                # print(weight)
                # print(predicted.max(),predicted.min(),weight.max(),weight.min())
                # bernoulli_kl = F.binary_cross_entropy(predicted,weight,reduction='none') * (mask*~onehot)
                # predicted = predicted.to(torch.float64)
                # weight = weight.to(torch.float64)
                bernoulli_kl = (
                    (
                        (weight) * torch.log(weight)
                        + (1 - weight) * torch.log(1 - weight)
                    )
                    - (
                        (weight) * torch.log(predicted)
                        + (1 - weight) * torch.log(1 - predicted)
                    )
                ) * (mask * (1 - onehot))
                # print("here", (bernoulli_kl>0).sum())
                # print((weight)*torch.log(weight) + (1-weight)*torch.log(1-weight))
                # print((bernoulli_kl<0).sum())
                bernoulli_kl = bernoulli_kl * (bernoulli_kl > 0)
                bernoulli_kl = bernoulli_kl.sum(-1)
                cross_entropy_true = -(
                    torch.gather(torch.log(predicted), -1, x0[:, :, None])
                ).squeeze(-1)
                nlog_p = bernoulli_kl + cross_entropy_true
                nlog_p = nlog_p * self.T  # T number of KL
                # print(bernoulli_kl.mean(-1),cross_entropy_true.mean(-1))
            elif self.config.eval.new_diff_calculate == "window":
                posterior = (
                    torch.ones_like(xt)
                    * torch.clamp(nominator / denominator, min=0.0, max=1.0)
                    * (xt > 0)
                )
                mask = one_hot_x0 > 0
                posterior = mask * torch.ones_like(xt) + (~mask) * posterior
                mc_samples = torch.distributions.Bernoulli(probs=posterior).sample()
                new_mask = mc_samples > 0
                model_prob = torch.exp(model_output)
                nlog_p = -torch.log((model_prob * new_mask).sum(-1))
                nlog_p = nlog_p * self.T  # T number of KL
            elif (
                self.config.eval.new_diff_calculate == "hybrid"
            ):  # implies a hybrid approach for t > 0.5 use full, < 0.5 use window
                posterior = (
                    torch.ones_like(xt)
                    * torch.clamp(nominator / denominator, min=0.0, max=1.0)
                    * (xt > 0)
                )
                mask = one_hot_x0 > 0
                posterior = mask * torch.ones_like(xt) + (~mask) * posterior
                mc_samples = torch.distributions.Bernoulli(probs=posterior).sample()
                new_mask = mc_samples > 0
                model_prob = torch.exp(model_output)
                nlog_p = -torch.log((model_prob * new_mask).sum(-1))
                window_nlog_p = nlog_p * self.T  # T number of KL

                mask = xt > 0
                predicted = torch.clamp(predicted, min=1e-6, max=1.0 - 1e-6)
                weight = torch.clamp(nominator / denominator, min=0.0, max=1.0 - 1e-6)
                onehot = F.one_hot(x0, num_classes=self.vocab_size)
                bernoulli_kl = (
                    (
                        (weight) * torch.log(weight)
                        + (1 - weight) * torch.log(1 - weight)
                    )
                    - (
                        (weight) * torch.log(predicted)
                        + (1 - weight) * torch.log(1 - predicted)
                    )
                ) * (mask * (1 - onehot))
                # bernoulli_kl = bernoulli_kl
                bernoulli_kl = bernoulli_kl.sum(-1)
                # print(bernoulli_kl)
                cross_entropy_true = -(
                    torch.gather(torch.log(predicted), -1, x0[:, :, None])
                ).squeeze(-1)
                nlog_p = bernoulli_kl + cross_entropy_true
                full_nlog_p = nlog_p * self.T  # T number of KL
                t_mask = t < 0.9
                # print(t_mask.shape,full_nlog_p.shape,window_nlog_p.shape)
                nlog_p = t_mask * full_nlog_p
                # +
                # (~t_mask) * window_nlog_p
            else:
                raise NotImplementedError
        # TODO: make an topk version, instead of top1
        # elif stage == "topk":
        #   top_index =
        else:
            raise NotImplementedError

        return nlog_p  # T number of KL

    def _forward_pass_diffusion(self, x0):
        t = self._sample_t(x0.shape[0], x0.device)
        if self.T > 0:
            t = (t * self.T).to(torch.int)
            t = t / self.T
            # t \in {1/T, 2/T, ..., 1}
            t += 1 / self.T

        if self.change_of_variables:
            unet_conditioning = t[:, None]
            f_T = torch.log1p(-torch.exp(-self.noise.sigma_max))
            f_0 = torch.log1p(-torch.exp(-self.noise.sigma_min))
            move_chance = torch.exp(f_0 + t * (f_T - f_0))
            move_chance = move_chance[:, None]
        else:
            sigma, dsigma = self.noise(t)
            unet_conditioning = sigma[:, None]
            move_chance = 1 - torch.exp(-sigma[:, None])

        xt = self.q_xt(x0, move_chance)
        model_output = self.forward(xt, unet_conditioning)
        utils.print_nans(model_output, "model_output")

        if self.parameterization == "sedd":
            return dsigma[:, None] * self._score_entropy(
                model_output, sigma[:, None], xt, x0
            )

        if self.T > 0:
            diffusion_loss = self._d3pm_loss(
                model_output=model_output, xt=xt, x0=x0, t=t
            )
            if self.parameterization == "d3pm":
                reconstruction_loss = self._reconstruction_loss(x0)
            elif self.parameterization == "subs":
                reconstruction_loss = 0
            elif self.parameterization == "dfm":
                reconstruction_loss = 0
            return reconstruction_loss + diffusion_loss

        # SUBS parameterization, continuous time.
        log_p_theta = torch.gather(
            input=model_output, dim=-1, index=x0[:, :, None]
        ).squeeze(-1)

        if self.change_of_variables or self.importance_sampling:
            return log_p_theta * torch.log1p(-torch.exp(-self.noise.sigma_min))

        return -log_p_theta * (dsigma / torch.expm1(sigma))[:, None]

    def _loss(self, x0, attention_mask):
        (input_tokens, output_tokens, attention_mask) = self._maybe_sub_sample(
            x0, attention_mask
        )

        if self.parameterization == "ar":
            logprobs = self.backbone(input_tokens, None)
            loss = -logprobs.gather(-1, output_tokens[:, :, None])[:, :, 0]
        elif self.parameterization == "bfn":
            loss = self._bfn_forward_loss(input_tokens, stage="training")
            if self.config.training.reconstruct_weight > 0:
                reconstruct_loss = self.compute_reconstruction_loss(
                    input_tokens, type=self.reconstruct_type
                )  # add reconstruct loss for bfn parameterization.
            else:
                reconstruct_loss = torch.zeros_like(loss)
            if (
                self.config.training.reconstruct_weight > 0
                and self.config.training.reconstruct_only
            ):  # a flag for only conduct recontruction over step t for finding beta.
                loss = reconstruct_loss
            else:
                loss = loss + self.config.training.reconstruct_weight * reconstruct_loss
        elif self.parameterization == "new_diff":
            loss = self._forward_new_diffusion(input_tokens, stage="training")
            reconstruct_loss = torch.zeros_like(loss)
        elif self.parameterization == "dfm":
            loss = self._forward_dfm(input_tokens, stage="training")
            reconstruct_loss = torch.zeros_like(loss)
        else:
            loss = self._forward_pass_diffusion(input_tokens)
            reconstruct_loss = torch.zeros_like(loss)

        nlls = loss * attention_mask
        count = attention_mask.sum()

        batch_nll = nlls.sum()
        token_nll = batch_nll / count

        reconstructs = reconstruct_loss * attention_mask
        batch_rec = reconstructs.sum()
        token_rec = batch_rec / count

        return Loss(
            loss=token_nll,
            nlls=nlls,
            token_mask=attention_mask,
            reconstruct=token_rec,
            rnlls=reconstructs,
        )

    def _dna_loss(self, x0, attention_mask):
        # (input_tokens, output_tokens,
        #  attention_mask) = self._maybe_sub_sample(
        #    x0, attention_mask)
        input_tokens = x0
        if self.parameterization == "ar":
            logprobs = self.backbone(input_tokens, None)
            loss = -logprobs.gather(-1, output_tokens[:, :, None])[:, :, 0]
        elif self.parameterization == "bfn":
            x0 = input_tokens[:, :, :4]
            signal = input_tokens[:, :, 4:5]
            loss = self._bfn_forward_loss(x0, signal, stage="training")
            if self.config.training.reconstruct_weight > 0:
                reconstruct_loss = self.compute_reconstruction_loss(
                    x0, signal, type=self.reconstruct_type, stage="training"
                )  # add reconstruct loss for bfn parameterization.
            else:
                reconstruct_loss = torch.zeros_like(loss)
            if (
                self.config.training.reconstruct_weight > 0
                and self.config.training.reconstruct_only
            ):  # a flag for only conduct recontruction over step t for finding beta.
                loss = reconstruct_loss
            else:
                loss = loss + self.config.training.reconstruct_weight * reconstruct_loss
        elif self.parameterization == "new_diff":
            loss = self._dna_forward_new_diffusion(input_tokens, stage="training")
            reconstruct_loss = torch.zeros_like(loss)
        else:
            loss = self._forward_pass_diffusion(input_tokens)

        # nlls = loss * attention_mask
        # count = attention_mask.sum()

        # batch_nll = nlls.sum()
        # token_nll = batch_nll / count

        # reconstructs = reconstruct_loss * attention_mask
        # batch_rec = reconstructs.sum()
        # token_rec = batch_rec / count

        return Loss(
            loss=loss.mean(), nlls=None, token_mask=None, reconstruct=None, rnlls=None
        )

    # def _reconstruct_only
    def _valid_loss(self, x0, attention_mask):
        (input_tokens, output_tokens, attention_mask) = self._maybe_sub_sample(
            x0, attention_mask
        )

        if self.parameterization == "ar":
            logprobs = self.backbone(input_tokens, None)
            loss = -logprobs.gather(-1, output_tokens[:, :, None])[:, :, 0]
        elif self.parameterization == "bfn":
            loss = self._bfn_forward_loss(input_tokens, stage="inference")
            reconstruct_loss = self.compute_reconstruction_loss(
                input_tokens, type=self.reconstruct_type
            )  # add reconstruct loss for bfn parameterization.
            # if self.config.training.reconstruct_weight > 0 and self.config.training.reconstruct_only:
            #   loss = reconstruct_loss
            # else:
            loss = loss + reconstruct_loss
        elif self.parameterization == "new_diff":
            loss = self._forward_new_diffusion(input_tokens, stage="inference")
            # reconstruct_loss = torch.zeros_like(loss)
            reconstruct_loss = self.compute_reconstruction_loss(
                input_tokens, type=self.reconstruct_type
            )
        else:
            loss = self._forward_pass_diffusion(input_tokens)
            reconstruct_loss = torch.zeros_like(loss)

        nlls = loss * attention_mask
        count = attention_mask.sum()

        batch_nll = nlls.sum()
        token_nll = batch_nll / count

        reconstructs = reconstruct_loss * attention_mask
        batch_rec = reconstructs.sum()
        token_rec = batch_rec / count

        return Loss(
            loss=token_nll,
            nlls=nlls,
            token_mask=attention_mask,
            reconstruct=token_rec,
            rnlls=reconstructs,
        )

    def _dna_valid_loss(self, x0, attention_mask):
        # (input_tokens, output_tokens,
        #  attention_mask) = self._maybe_sub_sample(
        #    x0, attention_mask)
        input_tokens = x0
        if self.parameterization == "ar":
            logprobs = self.backbone(input_tokens, None)
            loss = -logprobs.gather(-1, output_tokens[:, :, None])[:, :, 0]
        elif self.parameterization == "bfn":
            x0 = input_tokens[:, :, :4]
            signal = input_tokens[:, :, 4:5]
            loss = self._bfn_forward_loss(x0, signal, stage="inference")
            reconstruct_loss = self.compute_reconstruction_loss(
                x0, signal, type=self.reconstruct_type, stage="inference"
            )  # add reconstruct loss for bfn parameterization.
            # if self.config.training.reconstruct_weight > 0 and self.config.training.reconstruct_only:
            #   loss = reconstruct_loss
            # else:
            loss = loss + reconstruct_loss
        elif self.parameterization == "new_diff":
            loss = self._dna_forward_new_diffusion(input_tokens, stage="inference")
            reconstruct_loss = torch.zeros_like(loss)
        else:
            loss = self._forward_pass_diffusion(input_tokens)

        # nlls = loss * attention_mask
        # count = attention_mask.sum()

        # batch_nll = nlls.sum()
        # token_nll = batch_nll / count

        # reconstructs = reconstruct_loss * attention_mask
        # batch_rec = reconstructs.sum()
        # token_rec = batch_rec / count

        return Loss(
            loss=loss.mean(), nlls=None, token_mask=None, reconstruct=None, rnlls=None
        )

    def _score_entropy(self, log_score, sigma, xt, x0):
        """Computes the SEDD loss.

        Args:
          log_score: float torch.Tensor with shape (batch_size,
              diffusion_model_input_length, vocab_size),
              log score, output of the denoising network.
          xt: int torch.Tensor with shape (batch_size,
              diffusion_model_input_length), input.
          x0: int torch.Tensor with shape (batch_size,
              diffusion_model_input_length), input.
          sigma: float torch.Tensor with shape (batch_size, 1).

        Returns:
          loss with shape (batch_size, diffusion_model_input_length)
        """
        masked_indices = xt == self.mask_index

        expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
        q_ratio = 1 / expsig_minus_1[masked_indices]

        words_that_were_masked = x0[masked_indices]

        neg_term = q_ratio * torch.gather(
            log_score[masked_indices], -1, words_that_were_masked[..., None]
        ).squeeze(-1)
        score = log_score[masked_indices].exp()
        if self.mask_index == self.vocab_size - 1:
            pos_term = score[:, :-1].sum(dim=-1)
        else:
            pos_term = score[:, : self.mask_index].sum(dim=-1) + score[
                :, self.mask_index + 1 :
            ].sum(dim=-1)
        const = q_ratio * (q_ratio.log() - 1)

        entropy = torch.zeros(*xt.shape, device=xt.device)
        entropy[masked_indices] += pos_term - neg_term + const
        return entropy

    @torch.no_grad
    def sample_subs_guidance(self, n_samples, stride_length, num_strides, dt=0.001):
        ones = torch.ones(n_samples, dtype=self.dtype, device=self.device)

        num_steps = int(1 / dt)
        sampling_steps = 0
        intermediate_tokens = []
        target = None
        for _ in range(num_strides + 1):
            p_x0_cache = None
            x = self._sample_prior(n_samples, self.config.model.length).to(self.device)
            if target is not None:
                x[:, :-stride_length] = target
            for i in range(num_steps + 1):
                p_x0_cache, x_next = self._ddpm_caching_update(
                    x=x, t=(1 - i * dt) * ones, dt=dt, p_x0=p_x0_cache
                )
                if not torch.allclose(x_next, x) or self.time_conditioning:
                    p_x0_cache = None
                    sampling_steps += 1
                x = x_next
            x = self.forward(x, 0 * ones).argmax(dim=-1)
            intermediate_tokens.append(x[:, :stride_length].cpu().numpy())
            target = x[:, stride_length:]

        intermediate_tokens.append(target.cpu().numpy())
        intermediate_text_samples = []
        sequence_lengths = (
            (
                np.concatenate(intermediate_tokens, axis=1)[:, 1:]
                == self.tokenizer.eos_token_id
            ).cumsum(-1)
            == 0
        ).sum(-1)
        for i in range(2, len(intermediate_tokens) + 1):
            intermediate_text_samples.append(
                self.tokenizer.batch_decode(
                    np.concatenate(intermediate_tokens[:i], axis=1)
                )
            )
        return (sampling_steps, intermediate_text_samples, sequence_lengths)

    def restore_model_and_semi_ar_sample(self, stride_length, num_strides, dt=0.001):
        """Generate samples from the model."""
        # Lightning auto-casting is not working in this method for some reason
        if self.ema:
            self.ema.store(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )
            self.ema.copy_to(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )
        self.backbone.eval()
        self.noise.eval()
        (sampling_steps, samples, sequence_lengths) = self.sample_subs_guidance(
            n_samples=self.config.loader.eval_batch_size,
            stride_length=stride_length,
            num_strides=num_strides,
            dt=dt,
        )
        if self.ema:
            self.ema.restore(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )
        self.backbone.train()
        self.noise.train()
        return sampling_steps, samples, sequence_lengths


def check_sudoku_validity(sample_str):
    flat = np.array([int(c) for c in sample_str])
    board = flat.reshape(9, 9)

    violations = 0
    for i in range(9):
        for n in range(1, 10):
            violations += np.abs(np.count_nonzero(board[i, :] == n) - 1)
            violations += np.abs(np.count_nonzero(board[:, i] == n) - 1)

    for br in range(3):
        for bc in range(3):
            for n in range(1, 10):
                box = board[br * 3 : (br + 1) * 3, bc * 3 : (bc + 1) * 3]
                violations += np.abs(np.count_nonzero(box == n) - 1)

    return violations == 0, violations
