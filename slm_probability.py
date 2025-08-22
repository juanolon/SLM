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

import torch
import functools
from abc import abstractmethod

from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical as torch_Categorical
from torch.distributions.bernoulli import Bernoulli as torch_Bernoulli
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.uniform import Uniform

from math import log
from torch import Tensor


CONST_log_min = 1e-10

def safe_log(data: Tensor):
    return data.clamp(min=CONST_log_min).log()

class DiscreteDistribution:
    @property
    @abstractmethod
    def probs(self):
        pass

    @functools.cached_property
    def log_probs(self):
        return safe_log(self.probs)

    @functools.cached_property
    def mean(self):
        pass

    @functools.cached_property
    def mode(self):
        pass

    @abstractmethod
    def log_prob(self, x):
        pass

    @abstractmethod
    def sample(self):
        pass

class Bernoulli(DiscreteDistribution):
    def __init__(self, logits):
        self.bernoulli = torch_Bernoulli(logits=logits, validate_args=False)

    @functools.cached_property
    def probs(self):
        p = self.bernoulli.probs.unsqueeze(-1)
        return torch.cat([1 - p, p], -1)

    @functools.cached_property
    def mode(self):
        return self.bernoulli.mode

    def log_prob(self, x):
        return self.bernoulli.log_prob(x.float())

    def sample(self, sample_shape=torch.Size([])):
        return self.bernoulli.sample(sample_shape)


class Categorical(DiscreteDistribution):
    def __init__(self, logits):
        self.categorical = torch_Categorical(logits=logits, validate_args=False)
        self.n_classes = logits.size(-1)

    @functools.cached_property
    def probs(self):
        return self.categorical.probs

    @functools.cached_property
    def mode(self):
        return self.categorical.mode

    def log_prob(self, x):
        return self.categorical.log_prob(x)

    def sample(self, sample_shape=torch.Size([])):
        return self.categorical.sample(sample_shape)


class DiscreteDistributionFactory:
    @abstractmethod
    def get_dist(self, params: torch.Tensor, input_params=None, t=None) -> DiscreteDistribution:
        """Note: input_params and t are only required by PredDistToDataDistFactory."""
        pass


class BernoulliFactory(DiscreteDistributionFactory):
    def get_dist(self, params, input_params=None, t=None):
        return Bernoulli(logits=params.squeeze(-1))


class CategoricalFactory(DiscreteDistributionFactory):
    def get_dist(self, params, input_params=None, t=None):
        return Categorical(logits=params)