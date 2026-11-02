# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Memory Aware Synapses (MAS), ported from upstream aimagelab/mammoth.

Per-parameter importance is the accumulated sensitivity of the network's
*output* to each parameter -- the absolute gradient of the squared L2 norm of
the output, ``|d (1/2 ||f(x)||^2) / d theta|``, averaged over the task's data
(unsupervised: no labels). A quadratic penalty then anchors important
parameters to their end-of-task value. Online (EWC-style) accumulation across
tasks via ``gamma``.

Structure mirrors this clone's ``ewc_on`` / ``si`` (closed-form penalty grad
added in ``observe``). Baseline for the per-weight metaplasticity experiment
(HANDOFF_plasticity_mlp.md §3): MAS is output-sensitivity importance, the
head-to-head bar against the firing-consistency signal of ``plasticity_mlp``.
"""

import torch
import torch.nn as nn

from models.utils.continual_model import ContinualModel
from utils.args import ArgumentParser


class MAS(ContinualModel):
    """Continual learning via Memory Aware Synapses."""
    NAME = 'mas'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        parser.add_argument('--e_lambda', type=float, required=True,
                            help='lambda weight for the MAS penalty')
        parser.add_argument('--gamma', type=float, required=True,
                            help='gamma parameter for online (EWC-style) '
                                 'accumulation of importance across tasks')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super(MAS, self).__init__(backbone, loss, args, transform, dataset=dataset)

        self.checkpoint = None
        self.importance = None

    def penalty(self):
        if self.checkpoint is None:
            return torch.tensor(0.0).to(self.device)
        return self.args.e_lambda * (
            self.importance * ((self.net.get_params() - self.checkpoint) ** 2)).sum()

    @torch.enable_grad()
    def end_task(self, dataset):
        # Output-sensitivity importance: |grad of 1/2 ||f(x)||^2| over the task
        # data (label-free). Batched (sensitivity, not Fisher, so no per-example
        # loop is required).
        importance = torch.zeros_like(self.net.get_params())
        n_batches = 0
        for data in dataset.train_loader:
            inputs = data[0].to(self.device)
            self.opt.zero_grad()
            output = self.net(inputs)
            penalty = 0.5 * output.pow(2).sum(dim=1).mean()
            penalty.backward()
            importance += self.net.get_grads().abs()
            n_batches += 1
        importance /= max(n_batches, 1)

        if self.importance is None:
            self.importance = importance
        else:
            self.importance = self.args.gamma * self.importance + importance

        self.checkpoint = self.net.get_params().data.clone()

    def get_penalty_grads(self):
        return self.args.e_lambda * 2 * self.importance * (
            self.net.get_params().data - self.checkpoint)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        self.opt.zero_grad()
        outputs = self.net(inputs)
        if self.checkpoint is not None:
            self.net.set_grads(self.get_penalty_grads())
        loss = self.loss(outputs, labels)
        assert not torch.isnan(loss)
        loss.backward()
        # Bound the combined (task + closed-form penalty) gradient before the
        # step, mirroring this clone's sibling quadratic-penalty regularizer
        # `si` (models/si.py). MAS importance is the raw output-sensitivity
        # |grad| (no Fisher/probability normalization), so at e_lambda >= 1 the
        # explicit penalty step 2*e_lambda*importance*(theta-checkpoint)
        # overshoots the anchor and the weights diverge within a few steps of
        # task 2 (loss -> inf -> NaN -> the assert above kills the run, which is
        # the "higher-lambda MAS hangs/truncates" failure). Clipping each
        # weight's gradient to [-1, 1] caps the per-step move at lr and makes
        # every lambda stable; it is (near-)inert at the low-lambda e_lambda 0.1
        # baseline, where the gradient rarely reaches the clip.
        nn.utils.clip_grad.clip_grad_value_(self.get_parameters(), 1)
        self.opt.step()

        return loss.item()
