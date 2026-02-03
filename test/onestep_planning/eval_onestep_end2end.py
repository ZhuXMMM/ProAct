#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End2End one-step planning evaluation:

- Reconstruct a planning "state" from L1 frame-level labels at a given L2 sample idx
  (same indexing/order as L2 SlidingWindowDataset / LazySlidingWindowDataset).
- For each method:
    - Baseline: Entropy strategy using GT state (k5 / all configurable)
    - Model: Entropy strategy using model-predicted (trigger/task/future_steps)
  then simulate ONE robot decision + ONE human decision (hmin/switch/noswitch),
  and compute one-step metrics:
    - immediate_saved (0/1)
    - avg_entropy
    - thr_spread (global: entropy(thread_id over cross-thread detours) * detour_ratio)
    - cross_det (rate)
    - robot feasibility (prereq ok + optional conflict definition)
    - human idle/wait + reason split
    - human detour / switch indicators

This script is designed to consume:
  - our offline model outputs: outputs/onestep_planning/{original,ours}/*.jsonl
  - API two-stage outputs: outputs/onestep_planning/*/two_stage_eval_resume.jsonl
All those files use global "idx" as sample id.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm


# Ensure we import the intended task planner implementation.
# Priority:
#   1) test/onestep_planning/task_planner.py (user-copied, should match planning version)
#   2) test/planning/task_planner.py (fallback)
REPO_ROOT = Path(__file__).resolve().parents[2]
ONESTEP_DIR = REPO_ROOT / "test" / "onestep_planning"
PLANNING_DIR = REPO_ROOT / "test" / "planning"
for d in (ONESTEP_DIR, PLANNING_DIR):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from task_planner import TaskGraphManager, EntropyPlanner  # noqa: E402
import planning_llm_infer  # noqa: E402


# -----------------------------
# Data helpers (rebuild L2 indexing without images)
# -----------------------------
def load_vocabulary_from_annotation(annotation_path: str) -> Dict[int, str]:
    vocab: Dict[int, str] = {}
    ap = Path(annotation_path)
    if not ap.exists():
        return vocab
    try:
        with ap.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return vocab
    vocab_section = data.get("vocabulary") or {}
    if isinstance(vocab_section, dict):
        for k, v in vocab_section.items():
            try:
                idx = int(k)
            except Exception:
                continue
            if isinstance(v, str) and v.strip():
                vocab[idx] = v.strip()
    return vocab


@dataclass
class VideoRow:
    video_id: str
    frame_labels: List[int]
    frame_task_labels: List[int]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def load_l1_rows(l1_jsonl: Path) -> List[VideoRow]:
    rows = read_jsonl(l1_jsonl)
    out: List[VideoRow] = []
    for r in rows:
        vid = str(r.get("video_id", "") or "").strip()
        if not vid:
            continue
        # Skip UCF-Crime dataset (do not evaluate)
        if vid.startswith("UCF_CRIME_"):
            continue
        fl = r.get("frame_labels") or []
        ftl = r.get("frame_task_labels") or []
        if not isinstance(fl, list) or not fl:
            continue
        if not isinstance(ftl, list):
            ftl = []
        out.append(
            VideoRow(
                video_id=vid,
                frame_labels=[int(x) if isinstance(x, (int, float, str)) and str(x).strip() else 0 for x in fl],
                frame_task_labels=[int(x) if isinstance(x, (int, float, str)) and str(x).strip() else 0 for x in ftl],
            )
        )
    return out


@dataclass
class SampleMeta:
    video_id: str
    end: int  # index into frame_labels (same as L2 dataset "end")


def build_l2_index(rows: List[VideoRow], window_stride: int) -> List[SampleMeta]:
    metas: List[SampleMeta] = []
    stride = max(1, int(window_stride))
    for vr in rows:
        n = len(vr.frame_labels)
        for end in range(0, n, stride):
            metas.append(SampleMeta(video_id=vr.video_id, end=end))
    return metas


@dataclass
class Segment:
    step: str
    task: str
    start: int
    end: int  # inclusive


def compress_segments(
    frame_labels: List[int],
    frame_task_labels: List[int],
    vocab: Dict[int, str],
) -> List[Segment]:
    segs: List[Segment] = []
    last_lbl: Optional[int] = None
    start_idx: Optional[int] = None
    last_task_lbl: Optional[int] = None
    n = len(frame_labels)
    for i in range(n + 1):
        lbl = None
        task_lbl = None
        if i < n:
            try:
                lbl = int(frame_labels[i])
            except Exception:
                lbl = 0
            try:
                task_lbl = int(frame_task_labels[i]) if i < len(frame_task_labels) else 0
            except Exception:
                task_lbl = 0
        # treat non-positive as "no step"
        if i < n and (lbl is None or lbl <= 0):
            # boundary: close previous segment if any
            if last_lbl is not None and last_lbl > 0 and start_idx is not None:
                step_name = (vocab.get(last_lbl) or str(last_lbl)).strip()
                task_name = (vocab.get(last_task_lbl or 0) or (str(last_task_lbl) if last_task_lbl else "")).strip()
                segs.append(Segment(step=step_name, task=task_name, start=start_idx, end=i - 1))
                last_lbl = None
                start_idx = None
                last_task_lbl = None
            continue

        # close at end
        if i == n:
            if last_lbl is not None and last_lbl > 0 and start_idx is not None:
                step_name = (vocab.get(last_lbl) or str(last_lbl)).strip()
                task_name = (vocab.get(last_task_lbl or 0) or (str(last_task_lbl) if last_task_lbl else "")).strip()
                segs.append(Segment(step=step_name, task=task_name, start=start_idx, end=n - 1))
            break

        # new segment
        if last_lbl is None:
            last_lbl = lbl
            start_idx = i
            last_task_lbl = task_lbl
            continue

        # same label => continue segment
        if lbl == last_lbl:
            # keep task label from current frame (prefer last frame's task id)
            last_task_lbl = task_lbl
            continue

        # label changed: close prev and start new
        if last_lbl is not None and last_lbl > 0 and start_idx is not None:
            step_name = (vocab.get(last_lbl) or str(last_lbl)).strip()
            task_name = (vocab.get(last_task_lbl or 0) or (str(last_task_lbl) if last_task_lbl else "")).strip()
            segs.append(Segment(step=step_name, task=task_name, start=start_idx, end=i - 1))
        last_lbl = lbl
        start_idx = i
        last_task_lbl = task_lbl

    # remove empty names defensively
    segs = [s for s in segs if s.step]
    return segs


@dataclass
class PlanningState:
    idx: int
    video_id: str
    task_name: str
    completed_steps: List[str]
    human_remaining_gt: List[str]  # includes Terminate
    human_future_steps_gt: List[str]  # horizon-K
    gt_step_now: str


def build_state_for_sample(
    idx: int,
    vr: VideoRow,
    end: int,
    vocab: Dict[int, str],
    horizon: int,
    append_terminate: bool = True,
    precomputed_segs: Optional[List[Segment]] = None,
) -> Optional[PlanningState]:
    # determine GT trigger / current step and task at end
    if end < 0 or end >= len(vr.frame_labels):
        return None
    lbl = int(vr.frame_labels[end]) if end < len(vr.frame_labels) else 0
    if lbl <= 0:
        return None
    gt_step = (vocab.get(lbl) or str(lbl)).strip()
    task_lbl = int(vr.frame_task_labels[end]) if end < len(vr.frame_task_labels) else 0
    task_name = (vocab.get(task_lbl) or (str(task_lbl) if task_lbl > 0 else "")).strip()
    segs = precomputed_segs if precomputed_segs is not None else compress_segments(vr.frame_labels, vr.frame_task_labels, vocab)
    if not segs:
        return None
    # If task_name is missing, fall back to the segment task at this moment.
    if not task_name:
        for s in segs:
            if s.start <= end <= s.end:
                task_name = (s.task or "").strip()
                break
    if not task_name or task_name.lower() == "others":
        return None

    # find current segment by end index
    cur_i = None
    for i, s in enumerate(segs):
        if s.start <= end <= s.end:
            cur_i = i
            break
    if cur_i is None:
        # fallback: if we can't locate, still proceed with minimal state
        remaining = []
        if append_terminate:
            remaining.append("Terminate")
        return PlanningState(
            idx=idx,
            video_id=vr.video_id,
            task_name=task_name,
            completed_steps=[gt_step],
            human_remaining_gt=remaining,
            human_future_steps_gt=[],
            gt_step_now=gt_step,
        )

    # completed steps: all segments up to current whose task matches task_name
    completed: List[str] = []
    for s in segs[: cur_i + 1]:
        if (s.task or "").strip() == task_name:
            if not completed or completed[-1] != s.step:
                completed.append(s.step)
    if not completed or completed[-1] != gt_step:
        completed.append(gt_step)

    # remaining steps: segments after current whose task matches task_name, until task changes
    remaining_steps: List[str] = []
    for s in segs[cur_i + 1 :]:
        if (s.task or "").strip() != task_name:
            break
        remaining_steps.append(s.step)
    if append_terminate and (not remaining_steps or remaining_steps[-1].lower() != "terminate"):
        remaining_steps = list(remaining_steps) + ["Terminate"]

    future_k = list(remaining_steps[: max(0, int(horizon))]) if horizon and horizon > 0 else []
    return PlanningState(
        idx=idx,
        video_id=vr.video_id,
        task_name=task_name,
        completed_steps=completed,
        human_remaining_gt=remaining_steps,
        human_future_steps_gt=future_k,
        gt_step_now=gt_step,
    )


# -----------------------------
# Prediction parsing
# -----------------------------
@dataclass
class PredRecord:
    pred_is_trigger: bool
    pred_task: str
    pred_step: str
    pred_future_steps: List[str]


def _clean_step_name(x: Any) -> str:
    s = str(x or "").strip()
    if not s:
        return ""
    if s == "__SKIP_STAGE2__":
        return ""
    return s


def parse_pred_record(obj: Dict[str, Any]) -> PredRecord:
    # trigger
    if "pred_is_trigger" in obj:
        pred_is = bool(obj.get("pred_is_trigger"))
    else:
        try:
            pred_is = bool(int(obj.get("pred", 0)))
        except Exception:
            pred_is = False

    # task
    cand_task = str(obj.get("pred_task_matched", "") or "").strip() or str(obj.get("pred_task", "") or "").strip()
    if cand_task == "__SKIP_STAGE2__":
        cand_task = ""
    if cand_task.lower() == "others":
        cand_task = ""

    # current step (some outputs have pred_step / pred_step_matched)
    cand_step = str(obj.get("pred_step_matched", "") or "").strip() or str(obj.get("pred_step", "") or "").strip()
    if cand_step == "__SKIP_STAGE2__":
        cand_step = ""

    # future steps
    fs = obj.get("pred_future_steps")
    out_fs: List[str] = []
    if isinstance(fs, list):
        out_fs = [_clean_step_name(x) for x in fs if _clean_step_name(x)]
    # Some offline outputs might not fill pred_future_steps but can put a single step in pred_step
    if not out_fs:
        st = _clean_step_name(obj.get("pred_step", ""))
        if st:
            out_fs = [st]
    return PredRecord(pred_is_trigger=pred_is, pred_task=cand_task, pred_step=cand_step, pred_future_steps=out_fs)


def load_predictions(inputs: List[Path]) -> Dict[int, PredRecord]:
    by_idx: Dict[int, PredRecord] = {}
    for p in inputs:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                if "idx" not in obj:
                    continue
                try:
                    idx = int(obj.get("idx"))
                except Exception:
                    continue
                by_idx[idx] = parse_pred_record(obj)
    return by_idx


# -----------------------------
# LLM action selector (reuse planning_llm_infer prompt + parsing)
# -----------------------------
def _get_graph_text_for_prompt(graph: TaskGraphManager, max_lines: int = 400) -> str:
    """
    Match planning_llm_infer.py's "compact task graph" idea: list node id/name/parents/meta.
    """
    if graph is None:
        return "(no task graph)"
    lines: List[str] = []
    try:
        id2node = getattr(graph, "id2node", {}) or {}
        for nid, node in id2node.items():
            try:
                name = node.get("name", "")
                parents = node.get("parent_id")
                if parents is None:
                    parents = []
                if not isinstance(parents, list):
                    parents = [parents]
                pn = [id2node.get(str(p), {}).get("name", "") for p in parents]
                leaf = bool(node.get("is_leafnode", False))
                mid = bool(node.get("is_midlevel", False))
                cat = node.get("midlevel_category", "")
                lines.append(f"id:{nid} name:{name} p:{parents} pn:{pn} leaf:{leaf} mid:{mid} cat:{cat}")
            except Exception:
                continue
    except Exception:
        return "(no task graph)"
    return "\n".join(lines[: max(1, int(max_lines))])


def _load_llm_cache(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.exists():
        return {}
    out: Dict[int, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            try:
                idx = int(obj.get("idx"))
            except Exception:
                continue
            # Do NOT let a "bad" cache permanently block real API calls.
            # If a previous run wrote missing-api / dry-run records, ignore them on reload.
            reason = str(obj.get("reason", "") or "")
            if reason == "llm_dry_run_or_missing_api":
                continue
            out[idx] = obj
    return out


def _append_llm_cache(path: Path, rec: Dict[str, Any]) -> None:
    _ensure_parent(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def select_action_with_llm(
    *,
    idx: int,
    pred: PredRecord,
    gt_state: PlanningState,
    graph_for_prompt: TaskGraphManager,
    candidates: List[str],
    immediate_M: int,
    horizon: int,
    api_base: str,
    api_key: str,
    api_version: str,
    llm_model: str,
    transport: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    presence_penalty: Optional[float],
    cache_path: Optional[Path] = None,
    cache: Optional[Dict[int, Dict[str, Any]]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Returns a record dict with at least:
      {action, reason, confidence, candidate_actions, raw_response}
    Notes:
    - Enforces candidate membership via planning_llm_infer._extract_action_from_raw
    - Optionally resumes via cache_path (latest record per idx wins)
    """
    if cache is not None and idx in cache:
        obj = cache[idx]
        return {
            "idx": idx,
            "video_id": gt_state.video_id,
            "task_name_gt": gt_state.task_name,
            "task_name_pred": pred.pred_task,
            "action": str(obj.get("action", "Wait / None") or "Wait / None"),
            "reason": str(obj.get("reason", "cache_hit") or "cache_hit"),
            "confidence": obj.get("confidence", None),
            "candidate_actions": list(obj.get("candidate_actions") or candidates or []),
            "raw_response": str(obj.get("raw_response", "") or ""),
            "cache_hit": True,
        }

    # If not triggered or no task, don't call LLM.
    if (not pred.pred_is_trigger) or (not pred.pred_task):
        return {
            "idx": idx,
            "video_id": gt_state.video_id,
            "task_name_gt": gt_state.task_name,
            "task_name_pred": pred.pred_task,
            "action": "Wait / None",
            "reason": "no_trigger_or_no_task",
            "confidence": None,
            "candidate_actions": list(candidates or []),
            "raw_response": "",
            "cache_hit": False,
        }

    candidates = [c for c in (candidates or []) if isinstance(c, str) and c.strip()]
    # Match planning_llm_infer.py: robot is never allowed to choose Terminate (terminal state node).
    candidates = [c for c in candidates if str(c or "").strip().lower() != "terminate"]
    future_window = list(candidates[: max(0, int(horizon))]) if horizon and horizon > 0 else list(candidates)
    human_immediate = future_window[: max(0, int(immediate_M))]

    # No candidates => must wait.
    if not candidates:
        return {
            "idx": idx,
            "video_id": gt_state.video_id,
            "task_name_gt": gt_state.task_name,
            "task_name_pred": pred.pred_task,
            "action": "Wait / None",
            "reason": "no_candidates",
            "confidence": None,
            "candidate_actions": [],
            "raw_response": "",
            "cache_hit": False,
        }

    if dry_run or (not api_base) or (not api_key) or (not llm_model):
        return {
            "idx": idx,
            "video_id": gt_state.video_id,
            "task_name_gt": gt_state.task_name,
            "task_name_pred": pred.pred_task,
            "action": "Wait / None",
            "reason": "llm_dry_run_or_missing_api",
            "confidence": None,
            "candidate_actions": list(candidates or []),
            "raw_response": "",
            "cache_hit": False,
        }

    task_graph_text = _get_graph_text_for_prompt(graph_for_prompt)
    user_prompt = planning_llm_infer.USER_PROMPT_TEMPLATE.format(
        task_name=pred.pred_task,
        task_graph=task_graph_text,
        completed=", ".join(gt_state.completed_steps) if gt_state.completed_steps else "(none)",
        human_immediate=", ".join(human_immediate) if human_immediate else "(none)",
        human_future=", ".join(future_window) if future_window else "(none)",
        candidates=", ".join(candidates),
    )

    client = planning_llm_infer._build_client(
        api_base=api_base,
        api_key=api_key,
        api_version=api_version,
        model=llm_model,
        transport=transport,
    )
    raw = planning_llm_infer._call_chat(
        client=client,
        model=llm_model,
        # One-step eval uses the "no-parallel" prompt variant by default.
        # (Only keep basic constraints 1/2/6; do NOT add 3/4/5.)
        system_prompt=planning_llm_infer._build_system_prompt(parallel=False),
        user_prompt=user_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        presence_penalty=presence_penalty,
    )
    action, reason, conf_val = planning_llm_infer._extract_action_from_raw(raw, candidates)

    rec: Dict[str, Any] = {
        "idx": idx,
        "video_id": gt_state.video_id,
        "task_name_gt": gt_state.task_name,
        "task_name_pred": pred.pred_task,
        "action": action,
        "reason": reason,
        "confidence": conf_val,
        "candidate_actions": candidates,
        "raw_response": raw,
        "cache_hit": False,
    }
    return rec


def discover_jsonl_inputs(pred_path: Path) -> List[Path]:
    if pred_path.is_file():
        return [pred_path]
    if pred_path.is_dir():
        # IMPORTANT: pred directories also contain our generated logs (e.g., planning_actions.jsonl).
        # Only load *model prediction* JSONLs.
        # Priority:
        #   1) sharded rank outputs: epoch_*.rank*.jsonl / *.rank*.jsonl
        #   2) two-stage outputs: two_stage_eval_resume.jsonl
        #   3) fallback: *.jsonl excluding known generated logs
        rank_files = sorted(pred_path.glob("*.rank*.jsonl"))
        if rank_files:
            return rank_files
        two_stage = pred_path / "two_stage_eval_resume.jsonl"
        if two_stage.exists():
            return [two_stage]
        outs: List[Path] = []
        for p in sorted(pred_path.glob("*.jsonl")):
            name = p.name
            if name in {"planning_actions.jsonl", "llm_planning_actions.jsonl"}:
                continue
            outs.append(p)
        return outs
    return []


# -----------------------------
# One-step simulation + metrics
# -----------------------------
def _shannon_entropy_from_counts(counts: Dict[str, int]) -> float:
    tot = sum(int(v) for v in counts.values() if v and v > 0)
    if tot <= 0:
        return 0.0
    h = 0.0
    for v in counts.values():
        if not v or v <= 0:
            continue
        p = v / tot
        h -= p * math.log2(p)
    return float(h)


def _is_real_action_node(step: str, graph_env: Optional[TaskGraphManager] = None) -> bool:
    """Check if a step is a real action node (not midlevel, terminate, or wait)."""
    s = str(step or "").strip()
    if not s or s == "Wait / None":
        return False
    if s.lower() == "terminate":
        return False
    if graph_env is not None:
        node = (graph_env.name2node or {}).get(s)
        if not node:
            return False
        try:
            if bool(node.get("is_midlevel", False)):
                return False
        except Exception:
            pass
    return True


@dataclass
class OneStepOutcome:
    # robot
    robot_action_raw: str
    robot_action: str
    robot_entropy: float
    robot_prereq_ok: int
    robot_conflict_head: int
    robot_feasible: int
    robot_forced_wait_reason: str
    immediate_saved: int
    # APA (Average parallel action): whether robot acted in a different thread than previous human thread
    robot_parallel_action: int
    # human
    human_action: str
    human_idle: int
    idle_due_to_in_progress: int
    idle_due_to_prereq: int
    detour: int
    cross_detour: int
    switch: int
    cross_detour_thread: str
    # Whether the simulated human action is a "real action node" (exclude Terminate/midlevel/unknown).
    human_action_is_real: int


def simulate_one_step(
    *,
    state: PlanningState,
    graph_env: TaskGraphManager,
    robot_action: str,
    robot_entropy: Optional[float],
    human_mode: str,
    immediate_M: int = 1,
) -> OneStepOutcome:
    # Keep a stable copy for entropy computation (decision-time world state).
    completed0 = list(state.completed_steps)
    completed = list(state.completed_steps)
    remaining = list(state.human_remaining_gt)
    remaining = [s for s in remaining if s and s not in completed]
    head_before = remaining[0] if remaining else ""
    completed_before = list(completed)
    head_thread_before = graph_env.thread_map.get(head_before) if head_before else None
    head_enabled_before = False
    if head_before and str(head_before).strip().lower() != "terminate":
        try:
            node0 = graph_env.name2node.get(str(head_before))
            cond0 = node0.get("activation_condition", "TRUE") if node0 else "TRUE"
            head_enabled_before = bool(graph_env.check_condition(cond0, completed_before))
        except Exception:
            head_enabled_before = False

    def _is_real(step: str) -> bool:
        return bool(_is_real_action_node(str(step), graph_env))

    # Legal set at this state (align with planning_eval.py semantics):
    # - executable under task graph prerequisites
    # - not yet completed
    # NOTE: translated from Chinese
    human_immediate = remaining[:max(0, int(immediate_M))] if remaining else []
    try:
        legal_raw = graph_env.get_legal_robot_actions(completed, human_immediate)
        # Only allow real action nodes for robot (exclude midlevel/Terminate).
        legal_now = set([a for a in (legal_raw or []) if _is_real(str(a))])
    except Exception:
        legal_now = set()

    # robot prereq check (environment) — if not in task graph or prereq not met, force Wait / None
    action = robot_action or "Wait / None"
    action_raw = action
    if action.strip().lower() in {"wait", "none", "wait/none", "wait / none"}:
        action = "Wait / None"

    robot_prereq_ok = 0
    forced_wait_reason = ""
    if action != "Wait / None":
        node = graph_env.name2node.get(action)
        # If the action is not even a node in the (GT) task graph, it is invalid in this environment.
        if node is None:
            robot_prereq_ok = 0
            forced_wait_reason = "action_not_in_task_graph"
            action = "Wait / None"
        else:
            # Structural nodes (midlevel/Terminate) are not executable robot actions for our metrics.
            if (not _is_real(action)):
                robot_prereq_ok = 0
                forced_wait_reason = "not_action_node"
                action = "Wait / None"
                node = None
            else:
                cond = node.get("activation_condition", "TRUE")
                try:
                    robot_prereq_ok = 1 if graph_env.check_condition(cond, completed) else 0
                except Exception:
                    robot_prereq_ok = 0
                # Match planning_llm_infer semantics: if not executable, robot must wait.
                if robot_prereq_ok != 1:
                    forced_wait_reason = "prereq_not_met"
                    action = "Wait / None"
                # NOTE: translated from Chinese
                elif legal_now and action not in legal_now:
                    forced_wait_reason = "not_legal"
                    action = "Wait / None"

    # NOTE: translated from Chinese
    robot_conflict_head = 1 if (action != "Wait / None" and head_before and action == head_before) else 0
    robot_feasible = 1 if (action == "Wait / None" or (robot_prereq_ok == 1 and robot_conflict_head == 0)) else 0

    # immediate saved: robot completes a future GT step (in remaining) and wasn't already done, and prereq ok
    immediate_saved = 1 if (action != "Wait / None" and robot_prereq_ok == 1 and action in remaining and action not in completed) else 0

    # Apply robot action to world state if executable
    in_progress = action if (action != "Wait / None" and robot_prereq_ok == 1) else None
    if in_progress and in_progress not in completed:
        completed.append(in_progress)
    robot_hist: List[str] = []
    if in_progress:
        robot_hist.append(in_progress)
    # If robot completes something in remaining, remove it (for human decision)
    if in_progress and in_progress in remaining:
        try:
            remaining.remove(in_progress)
        except ValueError:
            pass
    # Remove head items already completed
    while remaining and remaining[0] in completed:
        remaining.pop(0)

    # Human decision
    human_choice: Optional[Tuple[int, str]] = None
    robot_preempted_head = bool(in_progress and head_before and (str(in_progress) == str(head_before)) and head_enabled_before)

    def _terminate_allowed_now() -> bool:
        # Match planning_eval.py: Terminate can only happen after all remaining real action steps are done.
        for s in (remaining or []):
            if str(s or "").strip().lower() == "terminate":
                continue
            if _is_real(str(s)) and (str(s) not in completed):
                return False
        return True

    def _enabled(step_name: str) -> bool:
        # Match planning_eval.py semantics:
        # - Terminate hard rule
        # - GT head is always allowed to execute (avoid deadlock due to missing prereqs)
        if not step_name or step_name in completed:
            return False
        if in_progress and step_name == in_progress:
            return False
        if str(step_name or "").strip().lower() == "terminate":
            return _terminate_allowed_now()
        if remaining and step_name == remaining[0]:
            return True
        node = graph_env.name2node.get(step_name)
        cond = node.get("activation_condition", "TRUE") if node else "TRUE"
        try:
            return bool(graph_env.check_condition(cond, completed))
        except Exception:
            return False

    human_mode = (human_mode or "hmin").strip().lower()
    if human_mode == "noswitch":
        if remaining:
            head = remaining[0]
            if _enabled(head):
                human_choice = (0, head)
    elif human_mode == "hmin":
        # Match planning_eval.py:
        # 1) Prefer head if enabled.
        if remaining:
            head = remaining[0]
            if _enabled(head):
                human_choice = (0, head)
        # 2) Otherwise choose an enabled step in a thread that robot touched least.
        if human_choice is None:
            robot_counts: Dict[str, int] = {}
            for a in robot_hist:
                tid = graph_env.thread_map.get(a) or "serial"
                robot_counts[str(tid)] = robot_counts.get(str(tid), 0) + 1
            best = None
            best_key = None
            for j, step in enumerate(remaining):
                if not _enabled(step):
                    continue
                tid = graph_env.thread_map.get(step) or "serial"
                key = (robot_counts.get(str(tid), 0), j)
                if best_key is None or key < best_key:
                    best_key = key
                    best = (j, step)
            human_choice = best
    elif human_mode == "switch":
        for j, step in enumerate(remaining):
            if _enabled(step):
                human_choice = (j, step)
                break
    else:
        raise ValueError(f"invalid human_mode={human_mode} (expected hmin|switch|noswitch)")

    human_idle = 0
    idle_due_to_in_progress = 0
    idle_due_to_prereq = 0
    detour = 0
    cross_detour = 0
    switch = 0
    cross_detour_thread = ""
    human_action = "Wait / None"
    human_action_is_real = 0

    if human_choice is None:
        human_idle = 1
        # Match planning_eval.py: if robot preempted the head in this tick, attribute idle to in_progress.
        if robot_preempted_head:
            idle_due_to_in_progress = 1
        else:
            idle_due_to_prereq = 1
    else:
        j, step = human_choice
        human_action = step
        human_action_is_real = 1 if _is_real(str(step)) else 0
        # detour stats w.r.t the current head (after robot)
        if j > 0:
            detour = 1
        cur_tid = graph_env.thread_map.get(step)
        # Cross/Thr attribution (match planning_eval.py):
        # robot-preempt-induced cross-thread detour only (independent of positional detour j>0).
        if robot_preempted_head and head_thread_before and cur_tid and str(cur_tid) != str(head_thread_before) and human_action_is_real:
            cross_detour = 1
            cross_detour_thread = str(cur_tid)

        # switch indicator: human-to-human thread switch relative to previous human step (gt_step_now).
        prev_tid = graph_env.thread_map.get(state.gt_step_now) if _is_real(str(state.gt_step_now)) else None
        cur_tid2 = graph_env.thread_map.get(step)
        if prev_tid and cur_tid2 and str(prev_tid) != str(cur_tid2) and human_action_is_real:
            switch = 1
        completed.append(step)

    # Entropy metric should be comparable across methods:
    # always compute it on the GT task graph and the *effective* action (after forcing to Wait).
    ent = 0.0
    if action != "Wait / None":
        try:
            ent = float(EntropyPlanner(graph_env).calculate_entropy(robot_history=[], candidate_action=action, human_history=completed0))
        except Exception:
            ent = 0.0

    # APA: compare robot action thread with previous human thread (gt_step_now).
    robot_parallel_action = 0
    try:
        prev_tid = graph_env.thread_map.get(state.gt_step_now) if _is_real(str(state.gt_step_now)) else None
        rob_tid = graph_env.thread_map.get(action) if _is_real(str(action)) else None
        if prev_tid and rob_tid and str(prev_tid) != str(rob_tid):
            robot_parallel_action = 1
    except Exception:
        robot_parallel_action = 0
    return OneStepOutcome(
        robot_action_raw=action_raw,
        robot_action=action,
        robot_entropy=ent,
        robot_prereq_ok=int(robot_prereq_ok),
        robot_conflict_head=int(robot_conflict_head),
        robot_feasible=int(robot_feasible),
        robot_forced_wait_reason=str(forced_wait_reason),
        immediate_saved=int(immediate_saved),
        robot_parallel_action=int(robot_parallel_action),
        human_action=human_action,
        human_idle=int(human_idle),
        idle_due_to_in_progress=int(idle_due_to_in_progress),
        idle_due_to_prereq=int(idle_due_to_prereq),
        detour=int(detour),
        cross_detour=int(cross_detour),
        switch=int(switch),
        cross_detour_thread=cross_detour_thread,
        human_action_is_real=int(human_action_is_real),
    )


def _select_min_entropy_action(
    *,
    graph_env: TaskGraphManager,
    completed_steps: List[str],
    candidates: List[str],
    immediate_M: int,
) -> Tuple[str, float]:
    """
    Strictly fair entropy selection for one-step eval:
    - Uses the SAME task graph as the environment/metrics (graph_env)
    - Uses the SAME history definition as metrics (human_history = completed_steps, robot_history = [])
    - Uses the SAME candidate list definition across methods (caller-provided candidates)
    Notes:
    - In the full simulation (`planning_eval.py`), "no抢活" is enforced at the LEGAL action stage
      via `get_legal_robot_actions(completed, human_immediate)`, and entropy selects from
      candidates ∩ legal. We follow the same approach in one-step eval, so this selector assumes
      candidates are already filtered by legality (including no-preemption).
    - Filters to executable actions in the environment (node exists + prereq ok).
    Returns (action, entropy_of_action). If no valid action exists, returns ("Wait / None", 0.0).
    """

    # Normalize/clean candidates (preserve order)
    cand0: List[str] = []
    seen = set()
    for x in candidates or []:
        s = str(x or "").strip()
        if not s or s == "Wait / None":
            continue
        if s in seen:
            continue
        seen.add(s)
        cand0.append(s)
    if not cand0:
        return "Wait / None", 0.0

    # Filter out too-early Terminate if any other candidate exists.
    if any(str(s).strip().lower() != "terminate" for s in cand0):
        cand0 = [s for s in cand0 if str(s).strip().lower() != "terminate"]
        if not cand0:
            return "Wait / None", 0.0

    # Environment-feasible candidates: must be a node and prereq ok at decision time.
    feasible: List[str] = []
    for a in cand0:
        node = graph_env.name2node.get(a)
        if node is None:
            continue
        cond = node.get("activation_condition", "TRUE")
        try:
            if graph_env.check_condition(cond, list(completed_steps)):
                feasible.append(a)
        except Exception:
            continue
    if not feasible:
        return "Wait / None", 0.0

    # Candidates should already be legal; keep selector purely entropy-based.
    pool = feasible

    planner = EntropyPlanner(graph_env)
    best_a = pool[0]
    best_e = float("inf")
    best_pos = 10**9
    for a in pool:
        try:
            e = float(planner.calculate_entropy(robot_history=[], candidate_action=a, human_history=list(completed_steps)))
        except Exception:
            e = 0.0
        # tie-break: earlier in original candidate order
        try:
            pos = cand0.index(a)
        except Exception:
            pos = 10**9
        if (e < best_e - 1e-9) or (abs(e - best_e) <= 1e-9 and pos < best_pos):
            best_e = e
            best_a = a
            best_pos = pos
    if best_a:
        return best_a, float(best_e if best_e != float("inf") else 0.0)
    return "Wait / None", 0.0


@dataclass
class Agg:
    n: int = 0
    # robot
    robot_nonwait: int = 0
    robot_prereq_ok: int = 0
    robot_feasible: int = 0
    robot_conflict_head: int = 0
    immediate_saved: int = 0
    robot_parallel_action: int = 0
    # Sum of entropy over effective (non-wait) robot actions.
    # This matches planning_eval.py's `Ent` definition (action-weighted).
    entropy_sum: float = 0.0
    # Legacy/debug: entropy sum only over samples where immediate_saved==1.
    entropy_sum_saved: float = 0.0
    # human
    human_exec: int = 0
    human_idle: int = 0
    idle_due_to_in_progress: int = 0
    idle_due_to_prereq: int = 0
    detour: int = 0
    cross_detour: int = 0
    switch: int = 0
    cross_detour_threads: Dict[str, int] = None

    def __post_init__(self) -> None:
        if self.cross_detour_threads is None:
            self.cross_detour_threads = {}

    def add(self, o: OneStepOutcome) -> None:
        self.n += 1
        if o.robot_action != "Wait / None":
            self.robot_nonwait += 1
        self.robot_prereq_ok += o.robot_prereq_ok
        self.robot_feasible += o.robot_feasible
        self.robot_conflict_head += o.robot_conflict_head
        self.immediate_saved += o.immediate_saved
        self.robot_parallel_action += int(getattr(o, "robot_parallel_action", 0))
        if o.robot_action != "Wait / None":
            self.entropy_sum += float(o.robot_entropy)
        if int(o.immediate_saved) == 1:
            self.entropy_sum_saved += float(o.robot_entropy)

        if o.human_idle:
            self.human_idle += 1
            self.idle_due_to_in_progress += o.idle_due_to_in_progress
            self.idle_due_to_prereq += o.idle_due_to_prereq
        else:
            # Match planning_eval.py: only count real human action nodes (exclude Terminate/midlevel).
            if int(getattr(o, "human_action_is_real", 0)) == 1:
                self.human_exec += 1
        self.detour += o.detour
        self.cross_detour += o.cross_detour
        self.switch += o.switch
        if o.cross_detour_thread:
            self.cross_detour_threads[o.cross_detour_thread] = self.cross_detour_threads.get(o.cross_detour_thread, 0) + 1

    def to_row(self, method: str) -> Dict[str, Any]:
        n = max(1, self.n)
        human_exec = max(1, self.human_exec)
        forced_h = _shannon_entropy_from_counts(self.cross_detour_threads)
        forced_ratio = (self.cross_detour / human_exec) if human_exec > 0 else 0.0
        thr_spread = float(forced_h * forced_ratio)
        immediate_saved_rate = self.immediate_saved / n
        # avg_entropy: action-weighted entropy over effective (non-wait) robot actions (planning_eval.py alignment).
        avg_entropy = (self.entropy_sum / self.robot_nonwait) if self.robot_nonwait > 0 else 0.0
        # Saved-conditioned entropy (legacy/debug).
        avg_entropy_saved = (self.entropy_sum_saved / self.immediate_saved) if self.immediate_saved > 0 else 0.0
        # avg over all samples (including robot waits as 0) for reference/debugging.
        avg_entropy_all = self.entropy_sum / n
        effective_avg_entropy = avg_entropy
        return {
            "method": method,
            "n_samples": self.n,
            "immediate_saved_rate": immediate_saved_rate,
            "effective_avg_entropy": effective_avg_entropy,
            "avg_entropy": avg_entropy,
            "avg_entropy_saved": avg_entropy_saved,
            "avg_entropy_all": avg_entropy_all,
            # APA: fraction of samples where robot action is in a different thread than previous human thread.
            "APA": self.robot_parallel_action / n,
            "thr_spread": thr_spread,
            # For paper-aligned interpretation, detour/cross/switch should be conditioned on
            # "human executed a step" (i.e., non-idle), consistent with thr_spread's denominator.
            "cross_det": self.cross_detour / human_exec,
            "detour": self.detour / human_exec,
            "human_switch": self.switch / human_exec,
            "human_idle": self.human_idle / n,
            "idle_due_to_in_progress": self.idle_due_to_in_progress / n,
            "idle_due_to_prereq": self.idle_due_to_prereq / n,
            "robot_action_rate": self.robot_nonwait / n,
            "robot_prereq_ok_rate": self.robot_prereq_ok / n,
            "robot_conflict_head_rate": self.robot_conflict_head / n,
            "robot_feasible_rate": self.robot_feasible / n,
        }


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_metrics_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    _ensure_parent(path)
    header = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# -----------------------------
# Main
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End2End one-step planning evaluation (one robot decision + one human decision)")
    p.add_argument("--pred_path", type=Path, required=True, help="Model output path (file or directory of .jsonl)")
    p.add_argument("--method_name", type=str, required=True, help="Method name for the model row (e.g., ours/original/qwen3-vl-flash)")
    # Repo-relative defaults (data is not included in this export; override via flags).
    p.add_argument("--l1_json", type=Path, default=Path("data/l1/l1_test.jsonl"))
    p.add_argument("--annotation", type=Path, default=Path("data/annotations/all_annotations.json"))
    p.add_argument("--window_stride", type=int, default=3, help="Must match L2 eval window_stride used to produce idx")
    p.add_argument("--horizon", type=int, default=5, help="Future horizon K (k5)")
    p.add_argument("--append_terminate", action="store_true", help="Append Terminate to remaining (recommended)")
    p.add_argument("--human_mode", type=str, default="hmin", choices=["hmin", "switch", "noswitch"])
    p.add_argument(
        "--action_selector",
        type=str,
        default="entropy",
        choices=["entropy", "llm"],
        help="How to select robot action given model outputs: entropy=EntropyPlanner; llm=call planning LLM API (planning_llm_infer prompt).",
    )
    p.add_argument(
        "--only_correct_task",
        action="store_true",
        help="If set, only evaluate samples where pred_task == GT task_name (skip wrong-task samples from denominator).",
    )
    p.add_argument("--entropy_candidate_mode", type=str, default="future", choices=["future", "remaining"], help="For entropy selector: use predicted futureK or predicted remaining (future is recommended).")
    # LLM selector args (OpenAI-compatible)
    p.add_argument("--llm_model", type=str, default="", help="LLM model name for planning (e.g., qwen3-vl-flash). Required when --action_selector llm")
    p.add_argument("--api_base", type=str, default="", help="OpenAI-compatible API base, e.g. https://api.xxx/v1")
    p.add_argument("--api_key", type=str, default="", help="API key")
    p.add_argument("--api_version", type=str, default="", help="Azure api_version if needed (otherwise keep empty)")
    # Defaults aligned with the main evaluation scripts (_get_env_sampling):
    #   temperature=0.7, top_p=0.8, presence_penalty=1.5
    # And use OpenAI SDK by default (same style as L2 eval backend=api).
    p.add_argument("--transport", type=str, default="sdk", choices=["http", "sdk"], help="planning_llm_infer transport")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.8)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--presence_penalty", type=float, default=1.5)
    p.add_argument("--immediate_M", type=int, default=1, help="HUMAN_IMMEDIATE_NEXT size for planning prompt")
    p.add_argument("--llm_cache", type=Path, default=None, help="Optional JSONL cache for LLM decisions (resume). Default: <out_dir>/llm_planning_actions.jsonl")
    p.add_argument("--llm_dry_run", action="store_true", help="Do not call API; always return Wait / None (for smoke test)")
    p.add_argument("--out_dir", type=Path, default=None, help="Output directory (default: pred_path dir)")
    p.add_argument("--metrics_subpath", type=str, default="metrics.csv", help="Where to write metrics.csv relative to out_dir")
    p.add_argument("--also_write_root_metrics", action="store_true", help="Also write <out_dir>/metrics.csv regardless of metrics_subpath")
    p.add_argument(
        "--action_log",
        type=Path,
        default=None,
        help="Unified per-sample action log JSONL path. Default: <out_dir>/planning_actions.jsonl (overwrites each run).",
    )
    p.add_argument(
        "--include_entropy_baseline",
        action="store_true",
        help="If set, also output an extra row for Entropy (GT oracle) as a baseline. Default: only output the requested method row.",
    )
    p.add_argument("--max_samples", type=int, default=None, help="Debug: limit number of evaluated GT-trigger samples")
    p.add_argument("--dump_per_sample", type=Path, default=None, help="Optional: dump per-sample outcomes JSONL")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pred_inputs = discover_jsonl_inputs(args.pred_path)
    if not pred_inputs:
        raise SystemExit(f"[ERROR] no jsonl found under {args.pred_path}")

    rows = load_l1_rows(args.l1_json)
    if not rows:
        raise SystemExit(f"[ERROR] failed to load l1 rows from {args.l1_json}")

    metas = build_l2_index(rows, window_stride=args.window_stride)
    vocab = load_vocabulary_from_annotation(str(args.annotation))
    preds = load_predictions(pred_inputs)

    # per-task graph cache
    graph_cache: Dict[str, TaskGraphManager] = {}
    # per-video compressed segments cache (expensive if recomputed per idx)
    seg_cache: Dict[str, List[Segment]] = {}

    def _get_graph(task_name: str) -> Optional[TaskGraphManager]:
        t = (task_name or "").strip()
        if not t:
            return None
        if t not in graph_cache:
            try:
                graph_cache[t] = TaskGraphManager(str(args.annotation), t)
            except Exception:
                return None
        return graph_cache.get(t)

    # map video_id -> VideoRow for fast lookup
    by_vid: Dict[str, VideoRow] = {vr.video_id: vr for vr in rows}

    agg_model = Agg()
    agg_baseline = Agg()  # only used when include_entropy_baseline=true

    # LLM decision cache
    out_dir = args.out_dir or (args.pred_path if args.pred_path.is_dir() else args.pred_path.parent)
    out_dir = Path(out_dir)
    llm_cache_path: Optional[Path] = None
    llm_cache: Optional[Dict[int, Dict[str, Any]]] = None
    if args.action_selector == "llm":
        llm_cache_path = args.llm_cache or (out_dir / "llm_planning_actions.jsonl")
        llm_cache = _load_llm_cache(llm_cache_path) if llm_cache_path else {}

    # Unified per-sample action log (all modes)
    action_log_path = args.action_log or (out_dir / "planning_actions.jsonl")
    _ensure_parent(action_log_path)
    action_log_f = action_log_path.open("w", encoding="utf-8")

    dump_f = None
    if args.dump_per_sample:
        _ensure_parent(args.dump_per_sample)
        dump_f = args.dump_per_sample.open("w", encoding="utf-8")

    evaluated = 0
    pbar = tqdm(enumerate(metas), total=len(metas), desc=f"one-step eval ({args.method_name})", ncols=0)
    for idx, meta in pbar:
        llm_rec: Optional[Dict[str, Any]] = None
        vr = by_vid.get(meta.video_id)
        if vr is None:
            continue
        if meta.video_id not in seg_cache:
            seg_cache[meta.video_id] = compress_segments(vr.frame_labels, vr.frame_task_labels, vocab)
        st = build_state_for_sample(
            idx=idx,
            vr=vr,
            end=meta.end,
            vocab=vocab,
            horizon=int(args.horizon),
            append_terminate=bool(args.append_terminate),
            precomputed_segs=seg_cache.get(meta.video_id),
        )
        if st is None:
            continue
        # Require at least one remaining step besides Terminate, otherwise planning is ill-defined
        rem = [x for x in (st.human_remaining_gt or []) if x and x.strip() and x.strip().lower() != "terminate"]
        if not rem:
            continue

        graph_env = _get_graph(st.task_name)
        if graph_env is None:
            continue

        # Optional Entropy baseline (oracle uses GT task + GT future/remaining)
        pred = preds.get(idx)
        if pred is None:
            # no prediction record => treat as no trigger (robot waits)
            pred = PredRecord(pred_is_trigger=False, pred_task="", pred_step="", pred_future_steps=[])

        # Fair comparison option: if task is predicted wrong, skip this sample entirely.
        # Rationale: if task is wrong, the predicted task graph cannot match GT graph,
        # and downstream step selection is not meaningful to compare.
        if bool(getattr(args, "only_correct_task", False)):
            if (pred.pred_task or "").strip() != (st.task_name or "").strip():
                continue

        if bool(getattr(args, "include_entropy_baseline", False)):
            # Match planning_eval.py: entropy selects from candidates ∩ legal (no-preemption enforced in legal).
            predicted_src0 = st.human_future_steps_gt if args.entropy_candidate_mode == "future" else st.human_remaining_gt
            predicted_src = [x for x in (predicted_src0 or []) if _is_real_action_node(str(x), graph_env)]
            human_immediate_gt = st.human_remaining_gt[:max(0, int(args.immediate_M))] if st.human_remaining_gt else []
            try:
                legal_raw = graph_env.get_legal_robot_actions(list(st.completed_steps), human_immediate_gt)
                legal_set = set([a for a in (legal_raw or []) if _is_real_action_node(str(a), graph_env)])
            except Exception:
                legal_set = set()
            future_for_entropy = [x for x in predicted_src if x in legal_set]
            a_base, e_base = _select_min_entropy_action(
                graph_env=graph_env,
                completed_steps=list(st.completed_steps),
                candidates=list(future_for_entropy or []),
                immediate_M=int(args.immediate_M),
            )
            o_base = simulate_one_step(
                state=st,
                graph_env=graph_env,
                robot_action=a_base,
                robot_entropy=e_base,
                human_mode=args.human_mode,
                immediate_M=int(args.immediate_M),
            )
            agg_baseline.add(o_base)

        # Method action selection
        a_model = "Wait / None"
        e_model = 0.0
        if args.action_selector == "entropy":
            # Strictly fair: use the same candidate definition as LLM (stage2 predicted future steps),
            # and compute entropy using the SAME GT task graph & history definition as metrics.
            if pred.pred_is_trigger and pred.pred_task:
                # Strategy candidates: predicted steps ∩ legal (environment-feasible).
                # Use same logic as planning_eval.py: immediate_M steps for legal check
                human_future_for_pred = pred.pred_future_steps if args.entropy_candidate_mode == "future" else pred.pred_future_steps
                human_immediate_gt = st.human_remaining_gt[:max(0, int(args.immediate_M))] if st.human_remaining_gt else []
                try:
                    legal_raw = graph_env.get_legal_robot_actions(list(st.completed_steps), human_immediate_gt)
                    # Only allow real action nodes for robot (exclude midlevel/Terminate).
                    legal_set = set([a for a in (legal_raw or []) if _is_real_action_node(str(a), graph_env)])
                except Exception:
                    legal_set = set()
                cand_inter = [c for c in (human_future_for_pred or []) if c in legal_set and _is_real_action_node(str(c), graph_env)]
                a_model, e_model = _select_min_entropy_action(
                    graph_env=graph_env,
                    completed_steps=list(st.completed_steps),
                    candidates=list(cand_inter),
                    immediate_M=int(args.immediate_M),
                )
        else:
            # LLM selector: candidates come from stage2 predicted future steps
            if pred.pred_is_trigger and pred.pred_task:
                graph_pred = _get_graph(pred.pred_task)
                if graph_pred is not None:
                    candidates = list(pred.pred_future_steps or [])
                    llm_rec = select_action_with_llm(
                        idx=idx,
                        pred=pred,
                        gt_state=st,
                        graph_for_prompt=graph_pred,
                        candidates=candidates,
                        immediate_M=int(args.immediate_M),
                        horizon=int(args.horizon),
                        api_base=str(args.api_base or ""),
                        api_key=str(args.api_key or ""),
                        api_version=str(args.api_version or ""),
                        llm_model=str(args.llm_model or ""),
                        transport=str(args.transport or "http"),
                        temperature=float(args.temperature),
                        top_p=float(args.top_p),
                        max_tokens=int(args.max_tokens),
                        presence_penalty=float(args.presence_penalty) if args.presence_penalty is not None else None,
                        cache_path=llm_cache_path,
                        cache=llm_cache,
                        dry_run=bool(getattr(args, "llm_dry_run", False)),
                    )
                    a_model = str(llm_rec.get("action", "Wait / None") or "Wait / None")
                    # Compute entropy for reporting (same definition as EntropyPlanner.calculate_entropy)
                    try:
                        e_model = float(
                            EntropyPlanner(graph_env).calculate_entropy(
                                robot_history=[],
                                candidate_action=a_model,
                                human_history=st.completed_steps,
                            )
                        )
                    except Exception:
                        e_model = 0.0

        o_model = simulate_one_step(
            state=st,
            graph_env=graph_env,
            robot_action=a_model,
            robot_entropy=e_model,
            human_mode=args.human_mode,
            immediate_M=int(args.immediate_M),
        )
        agg_model.add(o_model)

        # Unified per-sample action log record (all modes)
        try:
            rec_unified: Dict[str, Any] = {
                "idx": idx,
                "video_id": st.video_id,
                "method": str(args.method_name),
                "action_selector": str(args.action_selector),
                "human_mode": str(args.human_mode),
                "task_name_gt": st.task_name,
                "task_name_pred": pred.pred_task,
                "pred_is_trigger": bool(pred.pred_is_trigger),
                "human_now_gt": st.gt_step_now,
                "human_now_pred": pred.pred_step,
                "robot_action_raw": o_model.robot_action_raw,
                "robot_action_effective": o_model.robot_action,
                "robot_forced_wait_reason": o_model.robot_forced_wait_reason,
                "robot_prereq_ok": int(o_model.robot_prereq_ok),
                "robot_conflict_head": int(o_model.robot_conflict_head),
                "robot_feasible": int(o_model.robot_feasible),
                "robot_entropy": float(o_model.robot_entropy),
                "human_next_sim": o_model.human_action,
                "human_idle_sim": int(o_model.human_idle),
                "idle_due_to_in_progress_sim": int(o_model.idle_due_to_in_progress),
                "idle_due_to_prereq_sim": int(o_model.idle_due_to_prereq),
                "detour_sim": int(o_model.detour),
                "cross_detour_sim": int(o_model.cross_detour),
                "switch_sim": int(o_model.switch),
            }
            # Attach LLM metadata when applicable.
            if isinstance(llm_rec, dict) and args.action_selector == "llm":
                rec_unified.update(
                    {
                        "llm_action": llm_rec.get("action"),
                        "llm_reason": llm_rec.get("reason"),
                        "llm_confidence": llm_rec.get("confidence"),
                        "llm_cache_hit": bool(llm_rec.get("cache_hit", False)),
                        "llm_candidate_actions": llm_rec.get("candidate_actions"),
                        # Convenience: also include raw response in the unified log
                        # (full raw is always available in llm_planning_actions.jsonl).
                        "llm_raw_response": llm_rec.get("raw_response", ""),
                    }
                )
            action_log_f.write(json.dumps(rec_unified, ensure_ascii=False) + "\n")
        except Exception:
            pass

        # Enrich & append LLM action log (also acts as cache; last record per idx wins on reload).
        if args.action_selector == "llm" and llm_cache_path is not None:
            try:
                # Rebuild (or fetch) the record for this idx.
                # If we just called select_action_with_llm, llm_rec exists in local scope; otherwise rebuild from cache.
                rec0: Dict[str, Any]
                if "llm_rec" in locals() and isinstance(locals().get("llm_rec"), dict):
                    rec0 = dict(locals()["llm_rec"])
                else:
                    rec0 = dict((llm_cache or {}).get(idx, {}))
                enriched = dict(rec0)
                enriched.update(
                    {
                        "idx": idx,
                        "video_id": st.video_id,
                        "task_name_gt": st.task_name,
                        "task_name_pred": pred.pred_task,
                        # human "current" behavior
                        "human_now_gt": st.gt_step_now,
                        "human_now_pred": pred.pred_step,
                        # after applying (effective) robot action, simulated human next step
                        "robot_action_raw": o_model.robot_action_raw,
                        "robot_action_effective": o_model.robot_action,
                        "robot_forced_wait_reason": o_model.robot_forced_wait_reason,
                        "human_next_sim": o_model.human_action,
                        "human_idle_sim": int(o_model.human_idle),
                        "idle_due_to_in_progress_sim": int(o_model.idle_due_to_in_progress),
                        "idle_due_to_prereq_sim": int(o_model.idle_due_to_prereq),
                    }
                )
                _append_llm_cache(llm_cache_path, enriched)
                if llm_cache is not None:
                    llm_cache[idx] = enriched
            except Exception:
                pass

        if dump_f:
            dump_f.write(
                json.dumps(
                    {
                        "idx": idx,
                        "video_id": st.video_id,
                        "task_name": st.task_name,
                        "gt_step_now": st.gt_step_now,
                        "completed_steps": st.completed_steps,
                        "human_remaining_gt": st.human_remaining_gt,
                        "human_future_steps_gt": st.human_future_steps_gt,
                        "baseline": o_base.__dict__,
                        "model": o_model.__dict__,
                        "pred": {
                            "pred_is_trigger": pred.pred_is_trigger,
                            "pred_task": pred.pred_task,
                            "pred_future_steps": pred.pred_future_steps,
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        evaluated += 1
        pbar.set_postfix_str(f"evaluated={evaluated}")
        if isinstance(args.max_samples, int) and args.max_samples > 0 and evaluated >= args.max_samples:
            break
    pbar.close()
    action_log_f.close()

    if dump_f:
        dump_f.close()

    metrics_path = out_dir / args.metrics_subpath
    rows_out: List[Dict[str, Any]] = []
    if bool(getattr(args, "include_entropy_baseline", False)):
        rows_out.append(agg_baseline.to_row("Entropy"))
    rows_out.append(agg_model.to_row(str(args.method_name)))
    write_metrics_csv(metrics_path, rows_out)
    if args.also_write_root_metrics:
        # Also write a copy to <out_dir>/<basename(metrics_subpath)>.
        # This avoids overwriting legacy <out_dir>/metrics.csv when users want a suffixed metrics file (e.g., metrics_new.csv).
        root_name = Path(str(args.metrics_subpath)).name or "metrics.csv"
        root_metrics_path = out_dir / root_name
        if root_metrics_path.resolve() != metrics_path.resolve():
            write_metrics_csv(root_metrics_path, rows_out)

    print(f"[done] evaluated={evaluated}")
    print(f"[metrics] {metrics_path}")
    print(f"[actions] {action_log_path}")
    for r in rows_out:
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()

