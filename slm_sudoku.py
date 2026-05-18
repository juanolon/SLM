import itertools
import math
from dataclasses import dataclass
from slm_probability import CategoricalFactory
from slm_utils import DiscreteBayesianFlow, DiscreteBayesianFlowLoss

from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.loss import MixturePathGeneralizedKL

from tqdm.auto import tqdm
import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from torch import Tensor

import dataloader
import noise_schedule
import wandb
from models import dit_bfn


LOG2 = math.log(2)


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


@dataclass
class Loss:
    loss: torch.FloatTensor
    nlls: torch.FloatTensor
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


class Diffusion(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.vocab_size = 10

        self.sampler = self.config.sampling.predictor
        self.gen_ppl_eval_model_name_or_path = (
            self.config.eval.gen_ppl_eval_model_name_or_path
        )
        self.antithetic_sampling = self.config.training.antithetic_sampling
        self.importance_sampling = self.config.training.importance_sampling
        self.change_of_variables = self.config.training.change_of_variables
        self.reconstruct_type = self.config.training.reconstruct_type

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

        self.backbone = dit_bfn.BFN_DIT(self.config, vocab_size=self.vocab_size)

        self.T = self.config.T

        self.softplus = torch.nn.Softplus()
        # metrics are automatically reset at end of epoch
        metrics = torchmetrics.MetricCollection(
            {"nll": NLL(), "bpd": BPD(), "ppl": Perplexity()}
        )
        metrics.set_dtype(torch.float64)
        self.train_metrics = metrics.clone(prefix="train/")
        self.valid_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")
        self.prefix = ""

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
        self.neg_infinity = -1000000.0
        self.fast_forward_epochs = None
        self.fast_forward_batches = None
        self._validate_configuration()

        self.validation_batch_size = 32
        self.validation_sudoku_samples = []

        # debug
        for name, param in self.named_parameters():
            if not param.requires_grad:
                print(f"Non-trainable param: {name}, Shape: {param.shape}")

    def _validate_configuration(self):
        assert not (self.change_of_variables and self.importance_sampling)
        assert self.parameterization == "new_diff"
        assert self.config.eval.new_diff_calculate == "full"
        assert self.config.backbone == "dit_bfn"
        assert not self.config.subs_masking

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
        if self.trainer.global_rank == 0:
            wandb.watch(
                self.backbone,
                log="all",
                log_freq=100,     # every N steps
                log_graph=False,
            )
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

    def _process_sigma(self, sigma):
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

        with torch.amp.amp.autocast('cuda', dtype=torch.float32):
            logits = self.backbone(x, sigma)

        logits = self._new_diff_parameterization(
            logits, x, stage=stage
        )  # masked parameterization.
        # check whether sigma is t

        return logits

    def _compute_loss(self, batch, prefix):
        if prefix == "train":
            losses = self._loss(batch["answer"], batch["question"])
        elif prefix == "val" or prefix == "test":
            losses = self._valid_loss(batch["answer"], batch["question"])
        else:
            raise ValueError(f"Invalid prefix: {prefix}")

        seq_len = batch["answer"].shape[1]

        loss = losses.loss
        if prefix == "train":
            self.train_metrics.update(losses.nlls)
        elif prefix == "val":
            self.valid_metrics.update(losses.nlls)
        elif prefix == "test":
            self.test_metrics.update(losses.nlls)
        else:
            raise ValueError(f"Invalid prefix: {prefix}")
        rec_loss = losses.reconstruct

        return loss, rec_loss

    def on_train_epoch_start(self):
        self.backbone.train()
        self.noise.train()

    def training_step(self, batch, batch_idx):
        self.log("train/avg_rating", batch["rating"].mean(), on_step=True, on_epoch=False)

        loss, _ = self._compute_loss(batch, prefix="train")

        if torch.isnan(loss):
            print(f"[DEBUG] nan loss at step {self.global_step}")

        self.log(
            "trainer/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        # self.log_dict(self.train_metrics, on_step=False, on_epoch=True)

        return loss

    # def on_train_epoch_end(self):
    #     computed = self.train_metrics.compute()
    #     self.log_dict(computed, on_epoch=True)
    #     self.train_metrics.reset()

    def on_validation_epoch_start(self):
        if self.ema:
            self.ema.store(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )
            self.ema.copy_to(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )

        computed = self.train_metrics.compute()
        self.log_dict(computed, on_epoch=True, sync_dist=True)
        self.train_metrics.reset()

        self.backbone.eval()
        self.noise.eval()
        self.valid_metrics.reset()
        assert self.valid_metrics.nll.mean_value == 0
        assert self.valid_metrics.nll.weight == 0

    def validation_step(self, batch, batch_idx):
        loss, rec_loss = self._compute_loss(batch, prefix="val")

        # self.log_dict(
        #     {"val/loss": loss, "val/reconstruct_loss": rec_loss},
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        # )

        # self.log_dict(self.valid_metrics, on_step=False, on_epoch=True, sync_dist=True)

        # save a slice of sudoku samples from the validation split
        if batch_idx == 0 and len(self.validation_sudoku_samples) == 0:
            self.validation_sudoku_samples.append(
                (batch["question"][: self.validation_batch_size].detach().cpu(), batch["rating"][: self.validation_batch_size].detach().cpu())
            )

        return loss

    def on_validation_epoch_end(self):
        self.log_dict(
            self.valid_metrics.compute(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.valid_metrics.reset()

        if self.trainer.global_rank == 0:
            valid_count = 0
            total_violations = 0
            formatted_data = []

            questions, ratings = self.validation_sudoku_samples[0]
            board_batch = questions.to(self.device)
            batch_solutions = self._sample(board=board_batch, num_steps=100)

            for b_idx, solution in enumerate(batch_solutions):
                is_valid, violations, violations_idx = check_sudoku_validity(solution)
                if is_valid:
                    valid_count += 1
                total_violations += violations

                orig_board = board_batch[b_idx].tolist()
                solution_flat = solution.tolist()

                html_grid = "<div style='font-family: monospace; font-size: 16px; letter-spacing: 6px;'>"
                for i in range(81):
                    val = solution_flat[i]
                    orig_val = orig_board[i]

                    if orig_val != 0:
                        html_grid += f"<span style='color: #3b82f6; font-weight: bold;'>{val}</span>"
                    elif i in violations_idx:
                        html_grid += f"<span style='color: #ff4a4a; font-weight: bold;'>{val}</span>"
                    else:
                        html_grid += f"<span style='color: #a1a1aa;'>{val}</span>"

                    if (i + 1) % 9 == 0:
                        html_grid += "<br>"

                html_grid += "</div>"

                # solution_chars = [str(val) for val in solution.tolist()]
                # pretty_grid = "\n".join(
                #     ["".join(solution_chars[i : i + 9]) for i in range(0, 81, 9)]
                # )
                # formatted_data.append([pretty_grid, is_valid, violations])
                rating = ratings[b_idx].item()
                formatted_data.append([wandb.Html(html_grid), is_valid, violations, rating])

            num_evaluated = len(formatted_data)
            validity_rate = valid_count / num_evaluated
            avg_violations = total_violations / num_evaluated

            self.log_dict(
                {
                    "val/sudoku_validity_rate": validity_rate,
                    "val/avg_rule_violations": avg_violations,
                },
                sync_dist=False,
                rank_zero_only=True,
            )

            sampled_table = wandb.Table(
                columns=["Grid View", "Perfect", "Violations", "Rating"],
                data=formatted_data,
            )
            self.logger.experiment.log(
                {f"sudoku_samples_step_{self.global_step}": sampled_table},
                step=self.global_step
            )

        if self.ema:
            self.ema.restore(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )

    def on_test_epoch_start(self):
        self.test_metrics.reset()

    def on_test_epoch_end(self):
        test_dict = self.test_metrics.compute()
        self.log_dict(test_dict, on_step=False, on_epoch=True, sync_dist=True)

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

    def _sample_newdiff(self, bsz, numsteps, board=None):
        x_t = (
            torch.ones(bsz, self.config.sampling.length, self.vocab_size)
            / self.vocab_size
        ).to(self.device)  # [N, K] discrete prior

        if board is not None:
            puzzle_mask = (board == 0).unsqueeze(-1).to(self.device)
            board_onehot = (
                F.one_hot(board.long(), num_classes=self.vocab_size)
                .float()
                .to(self.device)
            )

            x_t = torch.where(puzzle_mask, x_t, board_onehot)

        for i in tqdm(range(1, numsteps + 1), desc="sampling"):
            t = torch.ones(bsz, 1).to(x_t.device) * (i - 1) / numsteps
            t = 1 - t  # reverse
            #   # [B,1,1]
            # print(f"x_t: {x_t.shape}, t: {t.shape}")
            # x_t: torch.Size([128, 81, 10]), t: torch.Size([128, 1])

            model_output = self.forward(x_t, t, stage="inference")
            # print(f"output t={i}, shape:{model_output.shape}:\n{model_output}")
            # output t=1000, shape:torch.Size([128, 81, 10]):
            # tensor([[[   -inf,    -inf,    -inf,  ...,  0.0000,    -inf,    -inf],

            model_output = torch.exp(model_output)
            # print("model output",model_output, t)

            t = t.unsqueeze(-1)

            nominator = torch.clamp(self.expected_nums(t - 1 / numsteps) - 1, min=1e-1)
            denominator = torch.clamp(self.expected_nums(t) - 1, min=1e-1)
            predicted = model_output * 1 + torch.clamp(
                nominator / denominator, max=1.0, min=0.0
            ) * (1 - model_output)
            predicted = torch.clamp(predicted, min=0.0, max=1.0)

            sample_pred = _sample_bernoulli(predicted) * (x_t > 0)
            sample_pred_sum = sample_pred.sum(-1, keepdim=True)
            mask = sample_pred_sum > 0

            sample_pred = torch.where(
                mask,
                sample_pred,
                F.one_hot(predicted.argmax(-1), num_classes=self.vocab_size),
            )
            x_t = sample_pred / torch.clamp(sample_pred.sum(-1, keepdim=True), min=0.0)

            if board is not None:
                x_t = torch.where(puzzle_mask, x_t, board_onehot)

        t = (torch.zeros(bsz, 1) + 1 / numsteps).to(x_t.device)
        predicted = self.forward(x_t, t, stage="inference")
        sample_pred = torch.argmax(predicted, dim=-1)

        # print(f"final: {sample_pred.shape}, {sample_pred}")
        # final: torch.Size([128, 81]), tensor([[7, 3, 4,  ..., 1, 9, 8],
        #         [5, 7, 2,  ..., 7, 9, 4],
        #         [4, 8, 1,  ..., 5, 6, 9]

        if board is not None:
            flat_mask = puzzle_mask.squeeze(-1).to(self.device)
            sample_pred = torch.where(flat_mask, sample_pred, board)

        return sample_pred

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
    def _sample(self, num_steps=None, board=None):
        """Generate samples from the model."""
        batch_size_per_gpu = self.validation_batch_size
        if num_steps is None:
            num_steps = self.config.sampling.steps

        return self._sample_newdiff(
            batch_size_per_gpu, num_steps, board
        )  # add new_diff sampler

    def _sample_t(self, n, device):
        _eps_t = torch.rand(n, device=device)
        if self.antithetic_sampling:
            offset = torch.arange(n, device=device) / n
            _eps_t = (_eps_t / n + offset) % 1
        t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
        if self.importance_sampling:
            return self.noise.importance_sampling_transformation(t)
        return t

    def _maybe_sub_sample(self, x0):
        seqlen = x0.shape[1]

        assert seqlen == self.config.model.length

        input_tokens = x0
        output_tokens = None

        return input_tokens, output_tokens

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

    def _forward_new_diffusion(self, x0, stage="training", question=None):
        t = self._sample_t(x0.shape[0], x0.device).unsqueeze(-1)  # [Batch, 1]

        t = (t * self.T).to(torch.int)
        t = t / self.T
        # t \in {1/T, 2/T, ..., 1}
        t += 1 / self.T

        x_t = self.get_xt_bernoulli(x0, t)  # [Batch, Seq, Features]
        # x_t = x_t / x_t.sum(-1,keepdim=True)
        # t = t.unsqueeze(-1)  # [B,1,1]
        # t = t.repeat(1, x_t.size(1), 1)

        if question is not None:
            hint_mask = (question > 0).unsqueeze(-1).to(self.device)
            x0_encoded = F.one_hot(x0.long(), num_classes=self.vocab_size).float().to(self.device)
            x_t = torch.where(hint_mask, x0_encoded, x_t)

        model_output = self.forward(x_t, t, stage=stage)
        # utils.print_nans(model_output, 'model_output')
        diffusion_loss = self._new_diffusion_loss(
            model_output=model_output, xt=x_t, x0=x0, t=t, stage=stage
        )
        return diffusion_loss

    def _new_diffusion_loss(self, model_output, xt, x0, t, stage="training"):
        mask = xt > 0

        n_t = self.expected_nums(t).unsqueeze(-1)

        n_t_1 = self.expected_nums(t - 1 / self.T).unsqueeze(-1)

        nominator = torch.clamp(n_t_1 - 1, min=1e-1)
        denominator = torch.clamp(n_t - 1, min=1e-1)

        # pred_theta(x^c_t)
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
                ((weight) * torch.log(weight) + (1 - weight) * torch.log(1 - weight))
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
        # TODO: make an topk version, instead of top1
        # elif stage == "topk":
        #   top_index =
        else:
            raise NotImplementedError

        return nlog_p  # T number of KL

    def _loss(self, x0, question=None):
        loss = self._forward_new_diffusion(x0, stage="training", question=question)

        seq_len = x0.shape[1]
        sum_loss = loss.sum(-1)
        return Loss(
            loss=loss.mean(),
            nlls=sum_loss / seq_len / self.T,
            reconstruct=torch.tensor(0.0, device=x0.device),
            rnlls=torch.zeros_like(sum_loss),
        )

    def _valid_loss(self, x0, question=None):
        loss = self._forward_new_diffusion(x0, stage="inference", question=question)

        reconstruct_loss = self.compute_reconstruction_loss(
            x0, type=self.reconstruct_type
        )

        seq_len = x0.shape[1]
        return Loss(
            loss=loss.mean(),
            nlls=loss.sum(-1) / seq_len,
            reconstruct=reconstruct_loss.mean(),
            rnlls=reconstruct_loss.flatten(),
        )


def check_sudoku_validity(sample_str):
    try:
        flat = np.array([int(c) for c in sample_str])
    except ValueError:
        return False, 1000, set(range(81))

    board = flat.reshape(9, 9)
    violations = 0
    violation_idx = set()

    for i in range(9):
        for n in range(1, 10):
            row_matches = np.where(board[i, :] == n)[0]
            violations += np.abs(len(row_matches) - 1)
            if len(row_matches) > 1:
                violation_idx.update(i * 9 + row_matches)

            col_matches = np.where(board[:, i] == n)[0]
            violations += np.abs(len(col_matches) - 1)
            if len(col_matches) > 1:
                violation_idx.update(col_matches * 9 + i)

    for br in range(3):
        for bc in range(3):
            for n in range(1, 10):
                box = board[br * 3 : (br + 1) * 3, bc * 3 : (bc + 1) * 3]
                tower_matches = np.where(box == n)
                violations += np.abs(len(tower_matches[0]) - 1)

                if len(tower_matches[0]) > 1:
                    for r, c in zip(*tower_matches):
                        violation_idx.add((br * 3 + r) * 9 + (bc * 3 + c))

    return violations == 0, violations, violation_idx
