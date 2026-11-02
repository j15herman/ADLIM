"""
TaskRouter: per-task sparse hippocampal buffers used for TEST-TIME selection.

Motivation (the step-6 diagnosis): the NCM oracle showed the backbone
representation separates old classes (Task-IL ~92%) but the single shared head
cannot read them out in Class-IL (~20%). The bottleneck is the readout, not the
features. So instead of (only) consolidating weights, give EACH TASK its own
projection + sparse buffer, learned/filled during that task, and at TEST TIME
route a sample to the task whose buffer it most resembles — converting the
strong Task-IL signal into Class-IL by inferring the task and biasing the
logits toward that task's class window.

Per task t we keep a SparseLayerBank (the existing fixed/learned projection ->
sparsify -> EMA "buffer" -> cosine match). Three projection modes (flag):

  - 'kwta_random': fixed-random projection + k-WTA, no training. Buffer = EMA of
    task-t codes. Routing relies on near-orthogonality of distinct codes.
  - 'l1':          learned projection, ReLU code, trained with an L1 sparsity
    penalty (projection rows renormalized each step to prevent the trivial
    collapse to zero). No labels.
  - 'supcon':      learned projection, k-WTA code, trained with supervised
    contrastive loss on task-t classes so codes cluster by class. Most
    discriminative for selection.

Routing is read-only at test time: encode the feature under each seen task's
projection, take cosine to that task's buffer, pick the argmax task, and add a
large uniform bias to that task's class-window logits. The bias is uniform
within any single window, so Task-IL (argmax restricted to the true window) is
left exactly unchanged; only the cross-window Class-IL argmax is affected.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sparse_bank import SparseLayerBank


class TaskRouter(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        proj_dim: int = 2000,
        mode: str = "kwta_random",
        sparsity_pct: float = 0.05,
        beta: float = 0.99,
        l1_weight: float = 1e-2,
        supcon_temp: float = 0.1,
        lr: float = 1e-3,
        eps: float = 1e-6,
        seed: int = 0,
    ):
        super().__init__()
        if mode not in ("kwta_random", "l1", "supcon"):
            raise ValueError(f"unknown route mode '{mode}'")
        self.feat_dim = feat_dim
        self.proj_dim = proj_dim
        self.mode = mode
        self.sparsity_pct = sparsity_pct
        self.beta = beta
        self.l1_weight = float(l1_weight)
        self.supcon_temp = float(supcon_temp)
        self.lr = lr
        self.eps = eps
        self._seed = seed

        self.banks = nn.ModuleList()          # one SparseLayerBank per task
        self.windows: List[Tuple[int, int]] = []
        self.optimizer: Optional[torch.optim.Optimizer] = None
        # Tracks the router's device so per-task banks created AFTER .to(device)
        # land on the right device.
        self.register_buffer("_dev", torch.zeros(1))

    @property
    def learned(self) -> bool:
        return self.mode in ("l1", "supcon")

    @property
    def n_tasks(self) -> int:
        return len(self.banks)

    # ------------------------------------------------------------------
    # Task lifecycle
    # ------------------------------------------------------------------

    def add_task(self, task_id: int, window: Tuple[int, int]) -> None:
        """Create the buffer+projection for a new task and point the optimizer
        at it (past tasks' projections are frozen)."""
        sparsity = "l1" if self.mode == "l1" else "kwta"
        bank = SparseLayerBank(
            in_dim=self.feat_dim, proj_dim=self.proj_dim, sparsity=sparsity,
            sparsity_pct=self.sparsity_pct, proj_learned=self.learned,
            beta=self.beta, eps=self.eps, seed=self._seed + task_id,
        )
        bank = bank.to(self._dev.device)
        self.banks.append(bank)
        self.windows.append((int(window[0]), int(window[1])))
        if self.learned:
            self.optimizer = torch.optim.Adam(
                self.banks[-1].proj.parameters(), lr=self.lr)
        else:
            self.optimizer = None

    def _cur(self) -> SparseLayerBank:
        return self.banks[-1]

    # ------------------------------------------------------------------
    # Training (during the current task)
    # ------------------------------------------------------------------

    def observe(self, feat: torch.Tensor, labels: torch.Tensor) -> Optional[float]:
        """One router training step on the CURRENT task: train its projection
        (supcon/l1; no-op for kwta_random) and EMA-update its buffer with the
        batch's codes. `feat` is the detached fc2 representation."""
        bank = self._cur()
        loss_val = None
        if self.learned and self.optimizer is not None:
            loss = (self._supcon_loss(bank, feat, labels) if self.mode == "supcon"
                    else self._l1_loss(bank, feat))
            if loss is not None:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                loss_val = float(loss.item())
                if self.mode == "l1":
                    self._renorm_rows(bank)   # keep ||proj rows|| = 1 (anti-collapse)
        with torch.no_grad():
            bank.update(bank.encode(feat))
        return loss_val

    def _l1_loss(self, bank: SparseLayerBank, feat: torch.Tensor) -> torch.Tensor:
        code = bank.encode(feat)                      # ReLU code, grad to proj
        return self.l1_weight * code.abs().mean()

    @torch.no_grad()
    def _renorm_rows(self, bank: SparseLayerBank) -> None:
        w = bank.proj.weight
        w.div_(w.norm(dim=1, keepdim=True) + self.eps)

    def _supcon_loss(self, bank: SparseLayerBank, feat: torch.Tensor,
                     labels: torch.Tensor) -> Optional[torch.Tensor]:
        """Supervised contrastive loss on the sparse codes (cosine logits).
        Returns None if no in-batch positive pair exists."""
        code = bank.encode(feat)                      # (B, P), grad to proj
        z = F.normalize(code, dim=-1, eps=self.eps)
        B = z.shape[0]
        if B < 2:
            return None
        logits = (z @ z.t()) / self.supcon_temp
        eye = torch.eye(B, dtype=torch.bool, device=z.device)
        logits = logits.masked_fill(eye, float("-inf"))
        same = (labels.view(-1, 1) == labels.view(1, -1)) & ~eye
        if not same.any():
            return None
        logits = logits - logits.max(dim=-1, keepdim=True).values.detach()
        exp = torch.exp(logits)
        denom = exp.sum(dim=-1).clamp_min(self.eps)
        pos = (exp * same).sum(dim=-1)
        has_pos = same.any(dim=-1)
        per_row = -torch.log(pos.clamp_min(self.eps) / denom)
        return per_row[has_pos].mean()

    # ------------------------------------------------------------------
    # Test-time routing
    # ------------------------------------------------------------------

    @torch.no_grad()
    def route(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict the task per sample by max buffer similarity.
        Returns (pred_task (B,), sims (B, n_tasks))."""
        sims = [bank.match(bank.encode(feat)) for bank in self.banks]
        S = torch.stack(sims, dim=1)                  # (B, T)
        return S.argmax(dim=1), S

    @torch.no_grad()
    def logit_bias(self, feat: torch.Tensor, n_classes: int,
                   bias: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Additive (B, n_classes) logit bias: +bias on each sample's predicted
        task window, 0 elsewhere. Uniform within a window, so Task-IL argmax is
        unaffected; only cross-window Class-IL argmax shifts."""
        pred, _ = self.route(feat)
        out = feat.new_zeros(feat.shape[0], n_classes)
        for t, (s, e) in enumerate(self.windows):
            sel = pred == t
            if bool(sel.any()):
                out[sel, s:min(e, n_classes)] += bias
        return out, pred
