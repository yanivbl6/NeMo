# Copyright (c) 2019 NVIDIA Corporation
import torch
import torch.nn as nn

from .parts.jasper import init_weights
from nemo.backends.pytorch.nm import LossNM, TrainableNM
from nemo.core.neural_types import *
from nemo.utils.decorators import add_port_docs


class CTCLossNM(LossNM):
    """
    Neural Module wrapper for pytorch's ctcloss
    Args:
        num_classes (int): Number of characters in ASR model's vocab/labels.
            This count should not include the CTC blank symbol.
        zero_infinity (bool): Whether to zero infinite losses and the associated gradients.
            By default, it is False. Infinite losses mainly occur when the inputs are too
            short to be aligned to the targets.
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            # "log_probs": NeuralType({1: AxisType(TimeTag), 0: AxisType(BatchTag), 2: AxisType(ChannelTag),}),
            # "targets": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            # "input_length": NeuralType({0: AxisType(BatchTag)}),
            # "target_length": NeuralType({0: AxisType(BatchTag)}),
            "log_probs": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "targets": NeuralType(('B', 'T'), LabelsType()),
            "input_length": NeuralType(tuple('B'), LengthsType()),
            "target_length": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        loss:
            NeuralType(None)
        """
        # return {"loss": NeuralType(None)}
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, num_classes, zero_infinity=False):
        super().__init__()

        self._blank = num_classes
        self._criterion = nn.CTCLoss(blank=self._blank, reduction='none', zero_infinity=zero_infinity)

    def _loss(self, log_probs, targets, input_length, target_length):
        input_length = input_length.long()
        target_length = target_length.long()
        targets = targets.long()
        loss = self._criterion(log_probs.transpose(1, 0), targets, input_length, target_length)
        # note that this is different from reduction = 'mean'
        # because we are not dividing by target lengths
        loss = torch.mean(loss)
        return loss

    def _loss_function(self, **kwargs):
        return self._loss(*(kwargs.values()))


class AngularSoftmaxLoss(LossNM):
    """
    Neural Module wrapper for pytorch's angular margin cosine softmax loss
    Args:
        num_classes (int): Number of characters in ASR model's vocab/labels.
            This count should not include the CTC blank symbol.
        zero_infinity (bool): Whether to zero infinite losses and the associated gradients.
            By default, it is False. Infinite losses mainly occur when the inputs are too
            short to be aligned to the targets.
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        return {
            # "log_probs": NeuralType({1: AxisType(TimeTag), 0: AxisType(BatchTag), 2: AxisType(ChannelTag),}),
            # "targets": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)}),
            # "input_length": NeuralType({0: AxisType(BatchTag)}),
            # "target_length": NeuralType({0: AxisType(BatchTag)}),
            "logits": NeuralType(('B', 'D'), LogitsType()),
            "targets": NeuralType(('B',), LabelsType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        loss:
            NeuralType(None)
        """
        # return {"loss": NeuralType(None)}
        return {"loss": NeuralType(elements_type=LossType())}

    def __init__(self, s=30.0, m=0.4):
        super().__init__()

        self.eps = 1e-7
        self.s = s
        self.m = m

    def _loss(self, logits, targets):
        # W = nn.functional.normalize(self.linear.weight, p=2, dim=1)
        # embs = nn.functional.normalize(embs, p=2, dim=1)

        # out = self.linear(embs)
        out = logits

        numerator = self.s * (torch.diagonal(out.transpose(0, 1)[targets]) - self.m)
        excl = torch.cat([torch.cat((out[i, :y], out[i, y + 1 :])).unsqueeze(0) for i, y in enumerate(targets)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)

    def _loss_function(self, **kwargs):
        return self._loss(*(kwargs.values()))
