"""Online Instance Matching (OIM) loss.

Introduced by Xiao et al., "Joint Detection and Identification Feature Learning
for Person Search" (CVPR 2017), and used by the PRW baseline. Implementation
adapted from SeqNet (https://github.com/serend1p1ty/SeqNet/blob/master/models/oim.py).

The loss keeps two non-parametric feature memories:
  * a look-up table (LUT) with one slot per labeled training identity, updated
    with a running average of the features of that identity;
  * a circular queue (CQ) with features of recent *unlabeled* persons, which act
    as extra negatives.

Each person embedding is classified against [LUT; CQ] via cosine similarity and
cross-entropy on the labeled entries. Because gradients flow only through the
input features (memories are buffers), training is stable even though each
identity appears in very few images per batch.
"""

import torch
import torch.nn.functional as F
from torch import autograd, nn


class _OIMFunction(autograd.Function):
    """Similarity against [LUT; CQ] with memory update in the backward pass.

    The memory update is done in backward() (not forward) so that it happens
    exactly once per optimization step, after the features have been used.
    The custom_fwd/custom_bwd decorators keep the whole function in fp32 even
    under autocast: the memories are fp32 buffers and must stay so.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(ctx, inputs, targets, lut, cq, cq_head, momentum):
        ctx.save_for_backward(inputs, targets)
        ctx.lut, ctx.cq, ctx.cq_head, ctx.momentum = lut, cq, cq_head, momentum
        return inputs.mm(torch.cat([lut, cq]).t())

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        inputs, targets = ctx.saved_tensors
        lut, cq, cq_head, momentum = ctx.lut, ctx.cq, ctx.cq_head, ctx.momentum

        grad_inputs = grad_output.mm(torch.cat([lut, cq])) if ctx.needs_input_grad[0] else None

        for feat, pid in zip(inputs, targets):
            if pid >= 0:  # labeled identity -> running-average LUT update
                lut[pid] = momentum * lut[pid] + (1.0 - momentum) * feat
                lut[pid] /= lut[pid].norm().clamp(min=1e-12)
            else:  # unlabeled person -> push into the circular queue
                cq[cq_head[0]] = feat
                cq_head[0] = (cq_head[0] + 1) % cq.size(0)

        return grad_inputs, None, None, None, None, None


class OIMLoss(nn.Module):
    """OIM loss over L2-normalized person embeddings.

    Args:
        num_features: embedding dimensionality.
        num_pids: number of labeled identities (LUT size); 483 for PRW train
            (the PRW paper says 482, but ID_train.mat lists 483).
        cq_size: circular queue size for unlabeled persons.
        momentum: LUT running-average momentum.
        scalar: inverse softmax temperature applied to the cosine similarities.
    """

    def __init__(self, num_features: int = 256, num_pids: int = 483,
                 cq_size: int = 500, momentum: float = 0.5, scalar: float = 30.0):
        super().__init__()
        self.momentum = momentum
        self.scalar = scalar
        self.register_buffer("lut", torch.zeros(num_pids, num_features))
        self.register_buffer("cq", torch.zeros(cq_size, num_features))
        self.register_buffer("cq_head", torch.zeros(1, dtype=torch.int64))

    def forward(self, embeddings: torch.Tensor, pids: torch.Tensor) -> torch.Tensor:
        """Compute the loss on person embeddings.

        Args:
            embeddings: [N, D] L2-normalized features of *person* proposals.
            pids: [N] identity labels in [0, num_pids) or -1 for unlabeled.
        """
        logits = _OIMFunction.apply(
            embeddings, pids, self.lut, self.cq, self.cq_head, self.momentum
        )
        # When all proposals are unlabeled (pids all -1), cross_entropy with
        # ignore_index=-1 returns NaN (0/0) in PyTorch ≥2.x. Return a
        # differentiable zero instead so the CQ is still updated via backward.
        if not (pids >= 0).any():
            return (logits * 0).sum()
        # Unlabeled persons still contribute as negatives (their features enter
        # the queue) but are excluded from the cross-entropy via ignore_index.
        return F.cross_entropy(logits * self.scalar, pids, ignore_index=-1)
