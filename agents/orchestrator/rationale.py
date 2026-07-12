"""Scientific Decision Graph — structured rationale for every modeling decision.

Each RationaleNode captures the 5 elements of a verifiable scientific decision:
  Decision · Evidence · Alternative · Confidence · Counterfactual
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RationaleNode:
    """A single verifiable decision in the SDM pipeline."""

    step: str           # e.g. "model_selection", "cv_strategy", "threshold"
    decision: str       # What was chosen, in plain language
    evidence: str       # Why — data/metrics supporting this choice
    alternative: str    # What was the runner-up and why it lost
    confidence: float   # 0-1, how confident the agent is in this decision
    counterfactual: str # What would happen if we chose differently

    # Optional structured data for rendering
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "decision": self.decision,
            "evidence": self.evidence,
            "alternative": self.alternative,
            "confidence": self.confidence,
            "counterfactual": self.counterfactual,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
        }

    def to_html(self) -> str:
        """Render as an HTML card for the evaluation report."""
        conf_pct = int(self.confidence * 100)
        conf_color = "#10b981" if conf_pct >= 70 else "#f59e0b" if conf_pct >= 40 else "#ef4444"
        metrics_html = ""
        if self.metrics:
            metrics_html = "<ul>" + "".join(
                f"<li><strong>{k}</strong>: {v}</li>" for k, v in self.metrics.items()
            ) + "</ul>"

        return f"""
<div class="sdg-node" style="
    border:1px solid #1e2d45; border-radius:14px; padding:16px; margin:10px 0;
    background: linear-gradient(135deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
">
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
        <span style="
            background: {conf_color}; color:#000; font-weight:700;
            padding:3px 10px; border-radius:99px; font-size:11px;
        ">{conf_pct}%</span>
        <strong style="font-size:14px;">{self.step}</strong>
    </div>
    <p style="margin:6px 0; color:#d1d5db;"><strong>决策:</strong> {self.decision}</p>
    <p style="margin:6px 0; color:#93c5fd;"><strong>证据:</strong> {self.evidence}</p>
    <p style="margin:6px 0; color:#6b7280;"><strong>备选:</strong> {self.alternative}</p>
    <p style="margin:6px 0; color:#f59e0b; font-style:italic;"><strong>反事实:</strong> {self.counterfactual}</p>
    {metrics_html}
</div>"""


class DecisionGraph:
    """Collects and renders all rationale nodes from a pipeline run."""

    def __init__(self):
        self.nodes: List[RationaleNode] = []

    def add(self, node: RationaleNode) -> None:
        self.nodes.append(node)

    def to_list(self) -> List[Dict[str, Any]]:
        return [n.to_dict() for n in self.nodes]

    def to_html(self) -> str:
        if not self.nodes:
            return "<p style='color:#6b7280;'>本运行未生成决策记录。</p>"
        cards = "".join(n.to_html() for n in self.nodes)
        return f"""
<div class="sdg-container">
    <h3 style="margin-bottom:12px;">Scientific Decision Graph ({len(self.nodes)} 个决策节点)</h3>
    {cards}
</div>"""

    def to_json(self) -> str:
        return json.dumps(self.to_list(), ensure_ascii=False, indent=2)


def make_rationale(
    step: str,
    decision: str,
    evidence: str,
    alternative: str = "未评估",
    confidence: float = 0.5,
    counterfactual: str = "未评估",
    metrics: Optional[Dict[str, Any]] = None,
) -> RationaleNode:
    """Factory for RationaleNode with sensible defaults."""
    from datetime import datetime
    return RationaleNode(
        step=step,
        decision=decision,
        evidence=evidence,
        alternative=alternative,
        confidence=min(1.0, max(0.0, confidence)),
        counterfactual=counterfactual,
        metrics=metrics or {},
        timestamp=datetime.now().isoformat(timespec="seconds"),
    )
