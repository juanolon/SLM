# Copyright 2023 NNAISENSE SA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file implements the Bayesian Flow and BFN loss for continuous and discrete variables.
Finally it implements the BFN using these objects.
For consistency we use always use a tuple to store input parameters.
It has just one element for discrete data (the probabilities) and two for continuous/discretized (mean & variance).
The probability distributions and network architectures are defined in probability.py and networks dir.
"Cts" is an abbreviation of "Continuous".
"""

import math
from abc import abstractmethod, ABC
from typing import Union, Optional

import torch
import torch.distributions as D
import torch.nn.functional as F
from torch import nn, Tensor

from slm_probability import (
    DiscreteDistributionFactory,
)
import numpy as np
# from utils_model import sandwich, float_to_idx

def sandwich(x: Tensor):
    return x.reshape(x.size(0), -1, x.size(-1))


def idx_to_float(idx: np.ndarray, num_bins: int):
    flt_zero_one = (idx + 0.5) / num_bins
    return (2.0 * flt_zero_one) - 1.0

def float_to_idx(flt: np.ndarray, num_bins: int):
    flt_zero_one = (flt / 2.0) + 0.5
    return torch.clamp(torch.floor(flt_zero_one * num_bins), min=0, max=num_bins - 1).long()

def generate_2d_tensor_with_ones(shape, num_ones):
    # Check for errors where the number of ones exceeds the columns
    if (num_ones > shape[1]).any():
        raise ValueError("num_ones contains values greater than the number of columns.")

    # Initialize a zero-filled tensor
    tensor = torch.zeros(shape, dtype=torch.int64).to(num_ones.device)
    
    # Create column indices for the last dimension
    col_indices = torch.arange(shape[1]).expand(len(num_ones), -1).to(num_ones.device)
    
    # Create a mask for valid elements based on num_ones
    mask = (col_indices < num_ones.unsqueeze(1)).float()
    
    # Shuffle the column indices to get random positions of "1"s
    shuffled_indices = torch.argsort(torch.rand(mask.size()).to(num_ones.device), dim=1)
    
    # Use the mask to scatter ones
    tensor = mask.gather(1, shuffled_indices).int()
    
    return tensor

class BayesianFlow(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_prior_input_params(self, data_shape: tuple, device: torch.device) -> tuple[Tensor, ...]:
        """Returns the initial input params (for a batch) at t=0. Used during sampling.
        For discrete data, the tuple has length 1 and contains the initial class probabilities.
        For continuous data, the tuple has length 2 and contains the mean and precision."""
        pass

    @abstractmethod
    def params_to_net_inputs(self, params: tuple[Tensor, ...]) -> Tensor:
        """Utility method to convert input distribution params to network inputs if needed."""
        pass

    @abstractmethod
    def get_alpha(self, i: Union[int, Tensor], n_steps: int) -> float:
        """Returns the alpha at step i of total n_steps according to the flow schedule. Used:
        a) during sampling, when i and alpha are the same for all samples in the batch.
        b) during discrete time loss computation, when i and alpha are different for samples in the batch."""
        pass

    @abstractmethod
    def get_sender_dist(self, x: Tensor, alpha: Union[float, Tensor], shape=torch.Size([])) -> D.Distribution:
        """Returns the sender distribution with accuracy alpha obtained by adding appropriate noise to the data x. Used:
        a) during sampling (same alpha for whole batch) to sample from the output distribution produced by the net.
        b) during discrete time loss computation when alpha are different for samples in the batch."""
        pass

    @abstractmethod
    def update_input_params(self, input_params: tuple[Tensor, ...], y: Tensor, alpha: float) -> tuple[Tensor, ...]:
        """Updates the distribution parameters using Bayes' theorem in light of noisy sample y.
        Used during sampling when alpha is the same for the whole batch."""
        pass

    @abstractmethod
    def forward(self, data: Tensor, t: Tensor) -> tuple[Tensor, ...]:
        """Returns a sample from the Bayesian Flow distribution over input parameters at time t conditioned on data.
        Used during training when t (and thus accuracies) are different for different samples in the batch.
        For discrete data, the returned tuple has length 1 and contains the class probabilities.
        For continuous data, the returned tuple has length 2 and contains the mean and precision."""
        pass


class Loss(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def cts_time_loss(self, data: Tensor, output_params: Tensor, input_params: Tensor, t: Tensor) -> Tensor:
        """Returns the continuous time KL loss (and any other losses) at time t (between 0 and 1).
        The input params are only used when the network is parameterized to predict the noise for continuous data."""
        pass

    @abstractmethod
    def discrete_time_loss(
        self, data: Tensor, output_params: Tensor, input_params: Tensor, t: Tensor, n_steps: int, n_samples: int = 20
    ) -> Tensor:
        """Returns the discrete time KL loss for n_steps total of communication at time t (between 0 and 1) using
        n_samples for Monte Carlo estimation of the discrete loss.
        The input params are only used when the network is parameterized to predict the noise for continuous data."""
        pass

    @abstractmethod
    def reconstruction_loss(self, data: Tensor, output_params: Tensor, input_params: Tensor) -> Tensor:
        """Returns the reconstruction loss, i.e. the final cost of transmitting clean data.
        The input params are only used when the network is parameterized to predict the noise for continuous data."""
        pass



# Discrete Data

def k_scheduler(t):
    return torch.exp(-0.64341721 * torch.exp(2*t) -3.01445058 * torch.exp(t) + 14.49771453).int() #scheduler the top nums to be keeped. 

def e_scheduler(t):
    return torch.exp(2.36965212 * torch.exp(2*t) -15.00923241 * torch.exp(t) +  23.70118847).int()

class DiscreteBayesianFlow(BayesianFlow):
    def __init__(
        self,
        n_classes: int,
        min_sqrt_beta: float = 1e-10,
        discretize: bool = False,
        epsilon: float = 1e-6,
        max_sqrt_beta: float = 1,
        binary_count: bool = False,
        onehot_sparse: bool = False,
        top_nums: int = -1,
        mc_samples: int = 50,
        linear_trans: float = 1.0,
        scheduler: str ="square", #beta= t^{2} beta1 
        uniform_encoded: bool = False,
        entropy_to_count: bool = False,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.min_sqrt_beta = min_sqrt_beta
        self.discretize = discretize
        self.epsilon = epsilon
        self.max_sqrt_beta = max_sqrt_beta
        self.uniform_entropy = math.log(self.n_classes)
        self.binary_count = binary_count
        self.onehot_sparse = onehot_sparse
        self.mc_samples = mc_samples
        self.top_nums = top_nums
        self.scheduler = scheduler
        self.linear_trans = linear_trans
        self.uniform_encoded = uniform_encoded
        self.entropy_to_count = entropy_to_count
        # print("------",self.n_classes)
    


    def t_to_sqrt_beta(self, t):
        if self.scheduler == "square":
            return  t * self.max_sqrt_beta 
        elif self.scheduler == "cubic":
            return (t**3).sqrt() * self.max_sqrt_beta 
        elif self.scheduler == "linear":
            return (t).sqrt() * self.max_sqrt_beta 
        elif self.scheduler == "c_s":
            return (3*t**2-2*t**3).sqrt() * self.max_sqrt_beta 
        elif self.scheduler == "exp": #add exponential 
            return ((torch.exp(t-1)-torch.exp(torch.tensor(-1.0)))/(torch.exp(torch.tensor(0.0))-torch.exp(torch.tensor(-1.0)))).sqrt() * self.max_sqrt_beta 
        elif self.scheduler == "cos":
            return (1-torch.cos(torch.pi*t/2)).sqrt() * self.max_sqrt_beta 
        else:
            raise NotImplementedError

    def count_dist(self, x, beta=None):
        # if len(x.shape)
        mean = (self.n_classes * F.one_hot(x.long(), self.n_classes)) - 1 #batch len k 
        # std_dev = math.sqrt(self.n_classes)
        diag_k = torch.eye(self.n_classes, device=x.device) * self.n_classes
        cov = diag_k - torch.ones_like(diag_k)
        cov = cov + torch.eye(self.n_classes,device=x.device) * 1e-3
        if beta is not None:
            mean = mean * beta #batch 1 1 
            # std_dev = std_dev * beta.sqrt()
            cov = cov.unsqueeze(0).unsqueeze(0).repeat(mean.shape[0],mean.shape[1],1,1)
            cov = cov * beta.unsqueeze(-1)
        
        return D.MultivariateNormal(mean, cov, validate_args=False)

    def count_sample(self, x, beta):
        return self.count_dist(x, beta).rsample()

    @torch.no_grad()
    def get_prior_input_params(self, data_shape: tuple, device: torch.device) -> tuple[Tensor]:
        return (torch.ones(*data_shape, self.n_classes, device=device) / self.n_classes,)

    @torch.no_grad()
    def params_to_net_inputs(self, params: tuple[Tensor],t=None) -> Tensor:
        params = params[0]
        if self.onehot_sparse:
            bs, seqlen, vocab = params.size()
            params = params.view(-1, vocab)
            params = torch.multinomial(input=params,num_samples=self.mc_samples,replacement=True)
            # print(params.size())
            params = params.view(bs, seqlen,self.mc_samples)
            params = F.one_hot(params,num_classes=self.n_classes).float().sum(-2)
            # probs = probs / probs.sum(dim=-1, keepdim=True)
            # params = (params > 0).float()  #do a onehot operation. 
            params = params / params.sum(dim=-1, keepdim=True) #normalized to sum as one.
        
        if self.top_nums > 0:
            _, topk_indices = torch.topk(params, k=self.top_nums, dim=-1, largest=True, sorted=False)
    
            # Step 2: Create a mask tensor initialized to zeros
            mask = torch.zeros_like(params, dtype=torch.bool)
            
            # Step 3: Scatter the top-k values to the mask
            mask.scatter_(dim=-1, index=topk_indices, value=True)
            
            # Step 4: Create the result tensor with top-k values and others set to 0
            result = torch.zeros_like(params)
            result[mask] = params[mask]
            params = result / result.sum(dim=-1, keepdim=True) 
        
        if self.uniform_encoded:
            assert t != None
            if len(t.shape) != 2:
                t = t.unsqueeze(-1)
            # print(t.shape) 
            keep_nums = e_scheduler(t) #Batch * 1, update to the 
            keep_nums = torch.clamp(keep_nums,max=(self.n_classes-1),min=1).long()
            # print(keep_nums)
            sort_params, _ = torch.sort(params,descending=True) #Batch * seq * vacab
            # print(sort_params.shape)
            thresholds = sort_params.gather(2, keep_nums.view(-1, 1, 1).expand(-1, sort_params.shape[1], 1))
            # print(thresholds.max())
            mask = (params > thresholds) 
            # print(mask)
            params[mask] = 1.0
            params[~mask] = 0.0 
            params = params / (params.sum(dim=-1,  keepdim=True)+1e-4)
        
        if self.entropy_to_count: #just do positional level expansion.
            assert self.uniform_encoded == False
            entropy = -(params * torch.log(params)).sum(-1) #Batch * seq
            entropy_to_count = torch.exp(entropy).clamp(max=(self.n_classes-1),min=1).int().long()
            # print(entropy_to_count.shape)
            sort_params, _ = torch.sort(params,descending=True) #Batch * seq * vacab
            thresholds = sort_params.gather(2, entropy_to_count.unsqueeze(-1))
            mask = (params > thresholds) 
            params[mask] = 1.0
            params[~mask] = 0.0 
            params = params / (params.sum(dim=-1,  keepdim=True)+1e-4)
       
        # print(params.shape)
        if self.n_classes == 2:
            params = params * 2 - 1  # We scale-shift here for MNIST instead of in the network like for text
            params = params[..., :1]
        return params
    
    def get_alpha_weight(self,t):
        beta_1 = (self.max_sqrt_beta) ** 2
        if self.scheduler == "square":
            return beta_1 * 2 * t * (self.linear_trans ** 2)
        elif self.scheduler == "linear":
            return beta_1 * self.linear_trans
        elif self.scheduler == "cubic":
            return beta_1 * (3 *t**2) *  (self.linear_trans ** 3)
        elif self.scheduler == 'c_s':
            if self.linear_trans != 1:
                raise NotImplementedError
            return beta_1 * (6*t - 6*t**2)
        elif self.scheduler == "exp":
            if self.linear_trans != 1:
                raise NotImplementedError
            return beta_1 * torch.exp(t-1) / (torch.exp(torch.tensor(0.0))-torch.exp(torch.tensor(-1.0)))
            # t_n = i/n_steps
            # t_n_1 = (i-1)/n_steps
            # return ((self.max_sqrt_beta) ** 2) * ((3*t_n**2-2*t_n**3)-(3*t_n_1**2-2*t_n_1**3))
        elif self.scheduler == 'cos':
            if self.linear_trans != 1:
                raise NotImplementedError
            return beta_1 * torch.pi/2 * torch.sin(torch.pi*t/2)
        else:
            raise NotImplementedError
    
    def _div_fn(self,t_n, t_n_1):
        if self.scheduler == "square":
            return t_n ** 2  - t_n_1 **2
        elif self.scheduler == 'linear':
            return t_n - t_n_1
        elif self.scheduler == 'cubic':
            return t_n **3 - t_n_1 ** 3
        elif self.scheduler == "exp":
            # return torch.exp(torch.tensor(t_n)-1) - torch.exp(torch.tensor(t_n_1)-1)
            return math.exp(t_n-1) - math.exp(t_n_1-1)
        elif self.scheduler == "cos":
            return (1-torch.cos(torch.pi* torch.tensor(t_n/2)) - (1-torch.cos(torch.pi*torch.tensor(t_n_1/2))))
        else:
            raise NotImplementedError


    def get_alpha(self, i: Union[int, Tensor], n_steps: int) -> Union[float, Tensor]:
        #TODO: I need to make this part 
        t_n = i/n_steps
        t_n_1 = (i-1)/n_steps
        beta1 = (self.max_sqrt_beta)**2

        return beta1 * self._div_fn(self.linear_trans*t_n,self.linear_trans*t_n_1)


    def get_sender_dist(self, x: Tensor, alpha: Union[float, Tensor], shape=torch.Size([])) -> D.Distribution:
        e_x = F.one_hot(x.long(), self.n_classes)
        alpha = alpha.unsqueeze(-1) if isinstance(alpha, Tensor) else alpha
        dist = D.Normal(alpha * ((self.n_classes * e_x) - 1), (self.n_classes * alpha) ** 0.5)
        return dist

    def update_input_params(self, input_params: tuple[Tensor], y: Tensor, alpha: float) -> tuple[Tensor]:
        new_input_params = input_params[0] * y.exp()
        new_input_params /= new_input_params.sum(-1, keepdims=True)
        return (new_input_params,)

    @torch.no_grad()
    def forward(self, data: Tensor, t: Tensor) -> tuple[Tensor]:
        if self.discretize:
            data = float_to_idx(data, self.n_classes)
        t = self.linear_trans * t #TODO: we do linear transformation here
        sqrt_beta = self.t_to_sqrt_beta(t.clamp(max=1 - self.epsilon))
        lo_beta = sqrt_beta < self.min_sqrt_beta
        sqrt_beta = sqrt_beta.clamp(min=self.min_sqrt_beta)
        beta = sqrt_beta.square().unsqueeze(-1)
        logits = self.count_sample(data, beta)
        probs = F.softmax(logits, -1)
        probs = torch.where(lo_beta.unsqueeze(-1), torch.ones_like(probs) / self.n_classes, probs)

        #do sampling
        # probs = F.one_hot(D.Categorical(probs=probs).sample(),num_classes=self.n_classes).float()  #sampling for the distribution. 

        if self.n_classes == 2:
            probs = probs[..., :1]
            probs = probs.reshape_as(data)
        input_params = (probs,)
        return input_params
        # psuedo_data = torch.argmax(probs,dim=-1)
        # print(psuedo_data.shape)

        # if self.binary_count:
        #     counts = (self.n_classes * (1-t)).int().squeeze(-1) # could be different for samples in the batch. [batsize]
        #     binary_vector = generate_2d_tensor_with_ones((probs.shape[0],probs.shape[2]),counts)
        #     binary_vector = binary_vector.unsqueeze(1).repeat(1,probs.shape[1],1)
        #     one_hot_flag =  F.one_hot(psuedo_data.long(), self.n_classes) > 0 
        #     binary_vector = torch.where(one_hot_flag, torch.ones_like(probs), binary_vector)
        #     probs = binary_vector / binary_vector.sum(dim=-1, keepdim=True)
        #     probs = torch.where(lo_beta.unsqueeze(-1), torch.ones_like(probs) / self.n_classes, probs)
        #     # print(probs.size())
        # else:
            #----------------gumbel operation
            # probs = F.gumbel_softmax(logits, tau=1.0, hard=False).float() #Gumbel training with sparse paramerruzation
            # probs = torch.topk(probs,k=5,dim=-1).indices
            # ----------------Top operation
            # # print(t)
            # counts = (self.n_classes * (1-t[0])).int().squeeze(-1).item()
            # # print(counts)
            # topk_values, topk_indices = torch.topk(probs, counts,  dim=-1)
            # mask = torch.zeros_like(probs)
            # mask.scatter_(-1, topk_indices, topk_values)
            # probs = F.softmax(mask, -1)
            #----------------sample operation
            # probs = F.one_hot(D.Categorical(probs=probs).sample()).float()
            #----------------nothing 
            # probs = probs
            #-------- multi-nomial
            # probs = D.Multinomial(total_count=50,probs=probs).sample()
            # probs = probs / probs.sum(dim=-1, keepdim=True)
            # print(probs)
        # probs = D.Multinomial(total_count=20,probs=probs).sample()
        # probs = probs / probs.sum(dim=-1, keepdim=True)
        


class DiscreteBayesianFlowLoss(Loss):
    def __init__(
        self,
        bayesian_flow: DiscreteBayesianFlow,
        distribution_factory: DiscreteDistributionFactory,
    ):
        super().__init__()
        self.bayesian_flow = bayesian_flow
        self.distribution_factory = distribution_factory
        self.K = self.bayesian_flow.n_classes
        # self.objective_func = torch.nn.CrossEntropyLoss()


    def cts_time_loss(self, data: Tensor, output_params: Tensor, input_params: Tensor, t,stage="training") -> Tensor:
        flat_output = sandwich(output_params)
        pred_probs = self.distribution_factory.get_dist(flat_output).probs
        flat_target = data.flatten(start_dim=1)
        if self.bayesian_flow.discretize:
            flat_target = float_to_idx(flat_target, self.K)
        # safe_mask =  flat_target ==0 or (flat_target> 0 and flat_target < self.K)
        # flat_target[~safe_mask] = 0.0 
        tgt_mean = torch.nn.functional.one_hot(flat_target.long(), self.K)
        kl = self.K * ((tgt_mean - pred_probs).square()).sum(-1)/2 #TODO: add the weight. 
        t = t.flatten(start_dim=1).float()
        alpha_t = self.bayesian_flow.get_alpha_weight(t)
        loss = alpha_t * kl
        return loss

    def discrete_time_loss(
        self, data: Tensor, output_params: Tensor, input_params: Tensor, t: Tensor, n_steps: int, n_samples=10
    ) -> Tensor:
        flat_target = data.flatten(start_dim=1)
        if self.bayesian_flow.discretize:
            flat_target = float_to_idx(flat_target, self.K)
        i = t * n_steps + 1
        alpha = self.bayesian_flow.get_alpha(i, n_steps).flatten(start_dim=1)
        sender_dist = self.bayesian_flow.get_sender_dist(flat_target, alpha)

        flat_output = sandwich(output_params)
        receiver_mix_wts = self.distribution_factory.get_dist(flat_output).probs
        receiver_mix_dist = D.Categorical(probs=receiver_mix_wts.unsqueeze(-2))
        classes = torch.arange(self.K, device=flat_target.device).long().unsqueeze(0).unsqueeze(0)

        receiver_components = self.bayesian_flow.get_sender_dist(classes, alpha.unsqueeze(-1))

        
        receiver_dist = D.MixtureSameFamily(receiver_mix_dist, receiver_components)

        y = sender_dist.sample(torch.Size([n_samples]))
        loss = n_steps * (sender_dist.log_prob(y) - receiver_dist.log_prob(y)).mean(0).sum(-1)
        # print(loss.shape)
        # .mean(1, keepdims=True)
        return loss

    def reconstruction_loss(self, data: Tensor, output_params: Tensor, input_params: Tensor) -> Tensor:
        flat_outputs = sandwich(output_params) # 
        flat_data = data.flatten(start_dim=1)  # [Bsz,seqlens]
        output_dist = self.distribution_factory.get_dist(flat_outputs) #param: [BSZ,seqlens,vocab_size]
        return -output_dist.log_prob(flat_data)
    
    # @torch.inference_mode()
    # def compute_reconstruction_loss(self, data: Tensor) -> Tensor:
    #     t = torch.ones_like(data).float()
    #     input_params = self.bayesian_flow(data, t)
    #     net_inputs = self.bayesian_flow.params_to_net_inputs(input_params)
    #     output_params: Tensor = self.net(net_inputs, t)
    #     return self.loss.reconstruction_loss(data, output_params, input_params).flatten(start_dim=1).mean()



class BFN(nn.Module):
    def __init__(self, net: nn.Module, bayesian_flow: BayesianFlow, loss: Loss):
        super().__init__()
        self.net = net
        self.bayesian_flow = bayesian_flow
        self.loss = loss

    @staticmethod
    @torch.no_grad()
    def sample_t(data: Tensor, n_steps: Optional[int]) -> Tensor:
        if n_steps == 0 or n_steps is None:
            t = torch.rand(data.size(0), device=data.device).unsqueeze(-1)
        else:
            t = torch.randint(0, n_steps, (data.size(0),), device=data.device).unsqueeze(-1) / n_steps
        t = (torch.ones_like(data).flatten(start_dim=1) * t).reshape_as(data)
        return t

    def forward(
        self, data: Tensor, t: Optional[Tensor] = None, n_steps: Optional[int] = None
    ) -> tuple[Tensor, dict[str, Tensor], Tensor, Tensor]:
        """
        Compute an MC estimate of the continuous (when n_steps=None or 0) or discrete time KL loss.
        t is sampled randomly if None. If t is not None, expect t.shape == data.shape.
        """

        t = self.sample_t(data, n_steps) if t is None else t
        # sample input parameter flow
        input_params = self.bayesian_flow(data, t)
        net_inputs = self.bayesian_flow.params_to_net_inputs(input_params,t)

        # compute output distribution parameters
        output_params: Tensor = self.net(net_inputs, t)

        # compute KL loss in float32
        with torch.autocast(device_type=data.device.type if data.device.type != "mps" else "cpu", enabled=False):
            if n_steps == 0 or n_steps is None:
                loss = self.loss.cts_time_loss(data, output_params.float(), input_params, t)
            else:
                loss = self.loss.discrete_time_loss(data, output_params.float(), input_params, t, n_steps)

        # loss shape is (batch_size, 1)
        return loss.mean()

    @torch.inference_mode()
    def compute_reconstruction_loss(self, data: Tensor,type="ce") -> Tensor: #could be l2/ce for cross entropy
        t = torch.ones_like(data).float()
        input_params = self.bayesian_flow(data, t)
        net_inputs = self.bayesian_flow.params_to_net_inputs(input_params,t)
        output_params: Tensor = self.net(net_inputs, t)
        if type == "ce":
            return self.loss.reconstruction_loss(data, output_params, input_params).flatten(start_dim=1).mean()
        elif type == "l2":
            return self.loss.cts_time_loss(data, output_params.float(), input_params, t)
        else:
            raise NotImplementedError

    @torch.inference_mode()
    def sample(self, data_shape: tuple, n_steps: int) -> Tensor:
        device = next(self.parameters()).device
        input_params = self.bayesian_flow.get_prior_input_params(data_shape, device)
        distribution_factory = self.loss.distribution_factory

        for i in range(1, n_steps + 1):
            t = torch.ones(*data_shape, device=device) * (i - 1) / n_steps
            output_params = self.net(self.bayesian_flow.params_to_net_inputs(input_params,t), t)
            output_sample = distribution_factory.get_dist(output_params, input_params, t).sample()
            output_sample = output_sample.reshape(*data_shape)
            alpha = self.bayesian_flow.get_alpha(i, n_steps)
            y = self.bayesian_flow.get_sender_dist(output_sample, alpha).sample()
            input_params = self.bayesian_flow.update_input_params(input_params, y, alpha)

        t = torch.ones(*data_shape, device=device)
        output_params = self.net(self.bayesian_flow.params_to_net_inputs(input_params,t), t)
        output_sample = distribution_factory.get_dist(output_params, input_params, t).mode
        output_sample = output_sample.reshape(*data_shape)
        return output_sample