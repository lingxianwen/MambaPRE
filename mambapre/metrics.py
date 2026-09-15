from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch

from .constants import length_bucket


@dataclass
class BoundaryAccumulator:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    exact_fields: int = 0
    predicted_fields: int = 0
    gold_fields: int = 0
    perfect_messages: int = 0
    messages: int = 0
    boundary_error_sum: float = 0.0
    predicted_cuts: int = 0
    role_correct: int = 0
    type_correct: int = 0
    role_bytes: int = 0
    type_bytes: int = 0

    @staticmethod
    def _spans(cuts: set[int], length: int) -> set[tuple[int, int]]:
        points = [0, *sorted(cuts), length]
        return {(start, end - start) for start, end in zip(points, points[1:])}

    def add(
        self,
        pred_cuts: set[int],
        gold_cuts: set[int],
        length: int,
        pred_roles: np.ndarray | None = None,
        gold_roles: np.ndarray | None = None,
        pred_types: np.ndarray | None = None,
        gold_types: np.ndarray | None = None,
    ) -> None:
        self.true_positive += len(pred_cuts & gold_cuts)
        self.false_positive += len(pred_cuts - gold_cuts)
        self.false_negative += len(gold_cuts - pred_cuts)
        pred_spans = self._spans(pred_cuts, length)
        gold_spans = self._spans(gold_cuts, length)
        self.exact_fields += len(pred_spans & gold_spans)
        self.predicted_fields += len(pred_spans)
        self.gold_fields += len(gold_spans)
        self.perfect_messages += int(pred_cuts == gold_cuts)
        self.messages += 1
        self.predicted_cuts += len(pred_cuts)
        if gold_cuts:
            self.boundary_error_sum += sum(
                min(abs(pred - gold) for gold in gold_cuts) for pred in pred_cuts
            )
        if pred_roles is not None and gold_roles is not None:
            valid_roles = gold_roles[:length] != -100
            self.role_correct += int(
                (pred_roles[:length][valid_roles] == gold_roles[:length][valid_roles]).sum()
            )
            self.role_bytes += int(valid_roles.sum())
        if pred_types is not None and gold_types is not None:
            valid_types = gold_types[:length] != -100
            self.type_correct += int(
                (pred_types[:length][valid_types] == gold_types[:length][valid_types]).sum()
            )
            self.type_bytes += int(valid_types.sum())

    @staticmethod
    def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return precision, recall, f1

    def summary(self) -> dict[str, float | int]:
        precision, recall, f1 = self._prf(
            self.true_positive, self.false_positive, self.false_negative
        )
        field_p, field_r, field_f1 = self._prf(
            self.exact_fields,
            self.predicted_fields - self.exact_fields,
            self.gold_fields - self.exact_fields,
        )
        result = {
            "n": self.messages,
            "boundary_precision": precision,
            "boundary_recall": recall,
            "boundary_f1": f1,
            "exact_field_precision": field_p,
            "exact_field_recall": field_r,
            "exact_field_f1": field_f1,
            "message_perfection": self.perfect_messages / self.messages if self.messages else 0.0,
            "avg_pred_to_gold_error": (
                self.boundary_error_sum / self.predicted_cuts if self.predicted_cuts else 0.0
            ),
        }
        if self.role_bytes:
            result["byte_role_accuracy"] = self.role_correct / self.role_bytes
            result["role_labeled_bytes"] = self.role_bytes
        if self.type_bytes:
            result["byte_type_accuracy"] = self.type_correct / self.type_bytes
            result["type_labeled_bytes"] = self.type_bytes
        return result


@dataclass
class StratifiedMetrics:
    overall: BoundaryAccumulator = field(default_factory=BoundaryAccumulator)
    by_protocol: dict[str, BoundaryAccumulator] = field(
        default_factory=lambda: defaultdict(BoundaryAccumulator)
    )
    by_length: dict[str, BoundaryAccumulator] = field(
        default_factory=lambda: defaultdict(BoundaryAccumulator)
    )

    def add(self, protocol: str, length: int, **kwargs) -> None:
        self.overall.add(length=length, **kwargs)
        self.by_protocol[protocol].add(length=length, **kwargs)
        self.by_length[length_bucket(length)].add(length=length, **kwargs)

    def summary(self) -> dict:
        return {
            "overall": self.overall.summary(),
            "by_protocol": {
                key: value.summary() for key, value in sorted(self.by_protocol.items())
            },
            "by_length": {
                key: value.summary() for key, value in sorted(self.by_length.items())
            },
        }


def evaluate_batch(
    accumulator: StratifiedMetrics,
    outputs: dict[str, torch.Tensor],
    batch: dict,
    threshold: float = 0.5,
) -> None:
    probabilities = torch.sigmoid(outputs["boundary_logits"]).detach().cpu().numpy()
    gold = batch["boundary_labels"].detach().cpu().numpy()
    roles = outputs.get("role_logits")
    types = outputs.get("type_logits")
    pred_roles = roles.argmax(dim=-1).detach().cpu().numpy() if roles is not None else None
    pred_types = types.argmax(dim=-1).detach().cpu().numpy() if types is not None else None
    gold_roles = batch["role_labels"].detach().cpu().numpy()
    gold_types = batch["type_labels"].detach().cpu().numpy()

    for idx, length_tensor in enumerate(batch["lengths"]):
        length = int(length_tensor)
        pred_cuts = set(np.flatnonzero(probabilities[idx, :length] >= threshold).tolist())
        gold_cuts = set(np.flatnonzero(gold[idx, :length] > 0.5).tolist())
        pred_cuts.discard(0)
        gold_cuts.discard(0)
        accumulator.add(
            protocol=batch["protocols"][idx],
            length=length,
            pred_cuts=pred_cuts,
            gold_cuts=gold_cuts,
            pred_roles=None if pred_roles is None else pred_roles[idx],
            gold_roles=None if pred_roles is None else gold_roles[idx],
            pred_types=None if pred_types is None else pred_types[idx],
            gold_types=None if pred_types is None else gold_types[idx],
        )
