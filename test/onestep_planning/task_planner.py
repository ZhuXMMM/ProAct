"""
基于 DAG 与线程熵最小化的机器人决策模块

包含两个核心类：
1. TaskGraphManager: 负责加载任务图、识别线程、判断动作合法性
2. EntropyPlanner: 负责计算线程熵并做出决策
"""

import json
import re
import math
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set, Any


def _as_bool(v: Any) -> bool:
    """Robust bool parsing for taxonomy fields that may be bool/int/str (e.g., 'true'/'false')."""
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        return v.strip().lower() in {"true", "1", "yes", "y", "t"}
    return False


class TaskGraphManager:
    """
    任务图管理器：负责加载 taxonomy、识别并行线程、判断动作合法性
    """
    
    def __init__(self, annotation_path: str, task_name: str, enable_or_fallback: bool = False):
        """
        Args:
            annotation_path: 包含 "vocabulary" 和 "taxonomy" 的大 JSON 路径
            task_name: 当前视频对应的任务名称 (e.g., "Assemble Bed")
            enable_or_fallback: 当 AND gate 无结果时是否退化到 OR 分支（默认关闭）
        """
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # NOTE: translated from Chinese
            taxonomy_section = data.get('taxonomy', {})
            if isinstance(taxonomy_section, dict):
                self.taxonomy = taxonomy_section.get(task_name, {})
            else:
                self.taxonomy = {}
        
        if not self.taxonomy:
            raise ValueError(f"Task '{task_name}' not found in taxonomy")
        
        # NOTE: translated from Chinese
        self.id2node = {str(k): v for k, v in self.taxonomy.items()}
        self.name2node = {v.get('name', ''): v for k, v in self.taxonomy.items() if 'name' in v}
        self.name2id = {v.get('name', ''): str(k) for k, v in self.taxonomy.items() if 'name' in v}
        
        # NOTE: translated from Chinese
        # NOTE: translated from Chinese (task)
        self.and_gates = self._build_and_gates()
        self.enable_or_fallback = bool(enable_or_fallback)
        # NOTE: translated from Chinese
        self.or_gates = self._build_or_gates()
        
        # NOTE: translated from Chinese (step)
        # NOTE: translated from Chinese
        self.thread_map = self._build_thread_map()
    
    def _build_thread_map(self) -> Dict[str, str]:
        """
        线程定义（与论文口径保持一致）：
        - 线程只在“真实 action 节点（可执行步骤）”上定义；midlevel/结构节点与 Terminate 不属于任何线程成员。
        - 线程由“指定的 midlevel 结构节点”诱导（midlevel_type == subtask / parallel）：
          - subtask: 根(0)下的每个 subtask start 诱导一个线程（以该 subtask 的 exclusive descendants 为线程成员）
          - parallel: 每个 parallel midlevel 节点的每个分支（其直接子节点）诱导一个线程（exclusive descendants）
        - 任何被多个分支可达的节点（merge/shortcut 等）回落到 serial_main。
        """
        thread_map: Dict[str, str] = {}

        # NOTE: translated from Chinese
        parent_to_children: Dict[str, List[str]] = {}
        for nid, node in self.id2node.items():
            pids = node.get("parent_id")
            if pids is None:
                continue
            if not isinstance(pids, list):
                pids = [pids]
            for pid in pids:
                parent_to_children.setdefault(str(pid), []).append(str(nid))

        def descendants(start_id: str) -> Set[str]:
            start_id = str(start_id)
            seen: Set[str] = set()
            stack = [start_id]
            while stack:
                cur = stack.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                for ch in parent_to_children.get(cur, []):
                    if ch not in seen:
                        stack.append(ch)
            return seen

        def anc_with_self(start_id: str) -> Set[str]:
            # NOTE: translated from Chinese (step)
            try:
                anc = self._get_ancestors(start_id)
            except Exception:
                anc = set()
            anc.add(str(start_id))
            return anc

        def is_real_action_id(nid: str) -> bool:
            node = self.id2node.get(str(nid), {})
            name = str(node.get("name", "") or "").strip()
            if not name:
                return False
            if name.lower() == "terminate":
                return False
            if _as_bool(node.get("is_midlevel", False)):
                return False
            if str(node.get("midlevel_type") or "").lower() == "parallel":
                return False
            return True

        def assign_ids(node_ids: Set[str], tid: str) -> None:
            for x in node_ids:
                node = self.id2node.get(str(x), {})
                name = node.get("name", "")
                if not name:
                    continue
                # NOTE: translated from Chinese
                if str(name).strip().lower() == "terminate":
                    continue
                if str(node.get("midlevel_type") or "").lower() == "parallel":
                    continue
                if _as_bool(node.get("is_midlevel", False)):
                    continue
                if name not in thread_map:
                    thread_map[name] = tid

        # NOTE: translated from Chinese
        # NOTE: translated from Chinese
        # NOTE: translated from Chinese
        subtask_roots: List[str] = []
        for nid, node in self.id2node.items():
            if str(node.get("midlevel_type") or "").lower() != "subtask":
                continue
            pid = node.get("parent_id")
            if pid is None:
                continue
            # NOTE: translated from Chinese
            if str(pid) == "0":
                subtask_roots.append(str(nid))
        sub_desc: Dict[str, Set[str]] = {rid: descendants(rid) for rid in subtask_roots}
        for rid in subtask_roots:
            others = set()
            for r2, ds2 in sub_desc.items():
                if r2 == rid:
                    continue
                others |= set(ds2)
            exclusive = set(sub_desc.get(rid, set())) - others
            assign_ids(exclusive, f"subtask_{rid}")

        # NOTE: translated from Chinese
        parallel_parents = [
            nid for nid, node in self.id2node.items() if node.get("midlevel_type") == "parallel"
        ]
        
        for p_id in parallel_parents:
            # NOTE: translated from Chinese
            # NOTE: translated from Chinese
            children = []
            for nid, node in self.id2node.items():
                pids = node.get('parent_id')
                if pids is None:
                    continue
                if not isinstance(pids, list):
                    pids = [pids]
                if str(p_id) in [str(p) for p in pids]:
                    children.append(nid)
            
            if not children:
                continue
            branch_desc: Dict[str, Set[str]] = {cid: (anc_with_self(cid) | descendants(cid)) for cid in children}
            common = set.intersection(*branch_desc.values()) if len(branch_desc) > 1 else set()
            for i, child_id in enumerate(children):
                tid = f"parallel_{p_id}_branch_{i}"
                assign_ids(set(branch_desc.get(child_id, set())) - common, tid)
        
        # NOTE: translated from Chinese
        for nid, node in self.id2node.items():
            if not is_real_action_id(nid):
                continue
            name = str(node.get("name", "") or "").strip()
            if name not in thread_map:
                thread_map[name] = "serial_main"
        
        return thread_map
    
    def _get_ancestors(self, start_id: str) -> Set[str]:
        """返回某个节点的所有祖先 id（不含自身）。"""
        start_id = str(start_id)
        visited: Set[str] = set()
        stack = [start_id]
        
        while stack:
            nid = stack.pop()
            node = self.id2node.get(str(nid))
            if not node:
                continue
            
            pids = node.get("parent_id")
            if pids is None:
                continue
            if not isinstance(pids, list):
                pids = [pids]
            
            for pid in pids:
                pid = str(pid)
                if pid not in visited:
                    visited.add(pid)
                    stack.append(pid)
        
        return visited
    
    def _build_and_gates(self) -> Dict[str, Dict[str, Any]]:
        """
        预解析所有 activation_condition 中带 AND 的节点，
        按「父节点 -> 该分支的叶子动作集合」的形式存起来。

        对于每个 AND gate，下列步骤用于确定每个必要父节点对应的叶子动作集合：

        1. 从 activation_condition 中解析所有参与 AND 的父节点 id（例如 (21) AND (22)）。
        2. 对每个父节点 p，向上遍历其祖先（包括自身），收集所有标记为 is_leafnode 的节点 id。
        3. 不同分支之间可能存在公共祖先叶节点（例如前置的“Place luggage”、“Open the suitcase”同时是多个分支的祖先）。
           这些公共叶节点代表各分支共用的准备动作，并不属于某个特定分支的独占任务。
           为避免在分支选择时误将这些公共动作作为分支剩余任务，需要将其从各分支的叶集合中移除。
        4. 构建字典结构 gates[gate_id]，记录每个父节点对应的分支叶集合。
        """
        gates: Dict[str, Dict[str, Any]] = {}
        
        for nid, node in self.id2node.items():
            cond = (node.get("activation_condition") or "").strip()
            if not re.search(r"\bAND\b", cond, flags=re.IGNORECASE):
                continue
            
            # NOTE: translated from Chinese
            ids = sorted(set(re.findall(r"\((\d+)\)", cond)), key=int)
            if len(ids) <= 1:
                continue
            
            branch_leafs: Dict[str, Set[str]] = {}
            
            for pid in ids:
                # NOTE: translated from Chinese
                anc = self._get_ancestors(pid)
                # NOTE: translated from Chinese
                anc.add(str(pid))
                
                # NOTE: translated from Chinese
                leaf_ids: Set[str] = {
                    x for x in anc
                    if _as_bool(self.id2node.get(str(x), {}).get("is_leafnode", False))
                }
                if not leaf_ids:
                    leaf_ids = set(anc)
                branch_leafs[str(pid)] = leaf_ids
            
            # NOTE: translated from Chinese
            if branch_leafs:
                # NOTE: translated from Chinese
                all_sets = list(branch_leafs.values())
                if len(all_sets) > 1:
                    common_leaves = set.intersection(*all_sets)
                else:
                    common_leaves = set()
                
                if common_leaves:
                    for pid in branch_leafs:
                        branch_leafs[pid] = branch_leafs[pid] - common_leaves
            
            gates[str(nid)] = {
                "gate_id": str(nid),
                "parent_ids": ids,
                "branch_leafs": branch_leafs,
            }
        
        return gates
    
    def _build_or_gates(self) -> Dict[str, Dict[str, Any]]:
        """
        预解析 activation_condition 中含 OR 的节点，便于在 AND 分支无结果时退化到 OR 选择：
        结构同 AND：记录 gate_id、parent_ids、branch_leafs。
        """
        gates: Dict[str, Dict[str, Any]] = {}
        
        for nid, node in self.id2node.items():
            cond = (node.get("activation_condition") or "").strip()
            if not re.search(r"\bOR\b", cond, flags=re.IGNORECASE):
                continue
            
            ids = sorted(set(re.findall(r"\((\d+)\)", cond)), key=int)
            if len(ids) <= 1:
                continue
            
            branch_leafs: Dict[str, Set[str]] = {}
            for pid in ids:
                anc = self._get_ancestors(pid)
                anc.add(str(pid))
                leaf_ids: Set[str] = {
                    x for x in anc
                    if _as_bool(self.id2node.get(str(x), {}).get("is_leafnode", False))
                }
                branch_leafs[str(pid)] = leaf_ids
            
            gates[str(nid)] = {
                "gate_id": str(nid),
                "parent_ids": ids,
                "branch_leafs": branch_leafs,
            }
        
        return gates
    
    def _find_and_gate_for_step(self, step_id: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        """
        给定当前步骤 id，找一个包含它的 AND gate：
        返回 (gate_id, 当前步骤所在分支的父节点 pid)，找不到返回 (None, None)。
        """
        if step_id is None:
            return None, None
        
        step_id = str(step_id)
        
        for gid, info in self.and_gates.items():
            for pid, leaf_ids in info["branch_leafs"].items():
                if step_id in leaf_ids:
                    return gid, pid
        
        return None, None
    
    def _find_or_gate_for_step(self, step_id: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        """
        给定当前步骤 id，找一个包含它的 OR gate：
        返回 (gate_id, 当前步骤所在分支的父节点 pid)，找不到返回 (None, None)。
        """
        if step_id is None:
            return None, None
        
        step_id = str(step_id)
        
        for gid, info in self.or_gates.items():
            for pid, leaf_ids in info["branch_leafs"].items():
                if step_id in leaf_ids:
                    return gid, pid
        
        return None, None
    
    def _canonicalize_list(self, raw_steps: Optional[List[str]]) -> List[str]:
        """把一个字符串列表直接匹配成 DAG 里的标准步骤名（不做模糊匹配）。"""
        if not raw_steps:
            return []
        mapped: List[str] = []
        
        for s in raw_steps:
            if isinstance(s, str) and s in self.name2id and s not in mapped:
                mapped.append(s)
        
        return mapped
    
    def check_condition(self, condition_str: str, completed_step_names: List[str]) -> bool:
        """
        解析逻辑表达式，如 "((1) AND (2)) OR (3)"
        
        Args:
            condition_str: 逻辑表达式字符串
            completed_step_names: 当前已完成的步骤名称列表
            
        Returns:
            bool: 条件是否满足
        """
        if not condition_str or condition_str == "TRUE" or condition_str.strip().upper() == "TRUE":
            return True
        
        # NOTE: translated from Chinese
        # NOTE: translated from Chinese
        # NOTE: translated from Chinese
        completed_ids: Set[str] = set()
        for name in completed_step_names:
            if name in self.name2id:
                completed_ids.add(str(self.name2id[name]))
        if completed_ids:
            closure: Set[str] = set(completed_ids)
            for cid in list(completed_ids):
                try:
                    closure |= self._get_ancestors(cid)
                except Exception:
                    continue
            completed_ids = closure
        
        # NOTE: translated from Chinese
        def replace_logic(match):
            nid = match.group(1)
            if nid in completed_ids:
                return "True"
            # NOTE: translated from Chinese
            # NOTE: translated from Chinese
            # NOTE: translated from Chinese
            node = self.id2node.get(str(nid))
            if node and _as_bool(node.get("is_midlevel", False)):
                return "True"
            return "False"
        
        # NOTE: translated from Chinese
        # NOTE: translated from Chinese
        eval_str = re.sub(r"\((\d+)\)", replace_logic, condition_str)
        # NOTE: translated from Chinese
        eval_str = re.sub(r"\bAND\b", "and", eval_str, flags=re.IGNORECASE)
        eval_str = re.sub(r"\bOR\b", "or", eval_str, flags=re.IGNORECASE)
        
        try:
            return bool(eval(eval_str))
        except Exception as e:
            # NOTE: translated from Chinese
            return False
    
    def _fuzzy_match_step_name(self, step_name: str, candidate_names: List[str]) -> Optional[str]:
        """
        使用模糊匹配找到最相似的步骤名称
        
        Args:
            step_name: 要匹配的步骤名称
            candidate_names: 候选名称列表
            
        Returns:
            最相似的名称，如果没有找到相似度 > 0.6 的则返回 None
        """
        if not step_name or not candidate_names:
            return None
        
        from difflib import SequenceMatcher
        step_lower = step_name.lower().strip()
        best_match = None
        best_score = 0.0
        
        for cand in candidate_names:
            if not cand:
                continue
            cand_lower = cand.lower().strip()
            # NOTE: translated from Chinese
            score = SequenceMatcher(None, step_lower, cand_lower).ratio()
            # NOTE: translated from Chinese
            if step_lower in cand_lower or cand_lower in step_lower:
                score = max(score, 0.8)
            if score > best_score:
                best_score = score
                best_match = cand
        
        # NOTE: translated from Chinese
        if best_score > 0.6:
            return best_match
        return None
    
    def get_legal_robot_actions(
        self, 
        current_completed_steps: List[str], 
        human_future_steps: List[str]
    ) -> List[str]:
        """
        获取机器人当前可以做的合法动作。
        
        说明：
        - 这里的 "legal" 仅表示 **环境可执行性**（未完成 + 前置条件满足）。
        - 不再把人类的下一步/未来步（human_future_steps）从 legal 中剔除；
          “避免抢活/不与人类 head 冲突”应由上层策略/候选集或评分规则处理，而不是由 legal 定义强行排除。
        
        Args:
            current_completed_steps: 当前已完成的步骤名称列表（支持模糊匹配）
            human_future_steps: (保留参数以兼容旧接口) 人类未来步骤列表；当前不会影响 legal 计算。
            
        Returns:
            List[str]: 合法的机器人动作列表
        """
        legal_actions = []
        
        # NOTE: translated from Chinese
        all_leaf_names = [node.get('name', '') for node in self.id2node.values() 
                         if _as_bool(node.get('is_leafnode', False)) and node.get('name', '')]
        
        # NOTE: translated from Chinese (step)
        mapped_completed = set()
        for step in current_completed_steps:
            if step in self.name2node:
                mapped_completed.add(step)
        
        for nid, node in self.id2node.items():
            name = node.get('name', '')
            if not name:
                continue
            # NOTE: translated from Chinese
            if node.get("midlevel_type") == "parallel":
                continue
            if _as_bool(node.get("is_midlevel", False)):
                continue
            
            # NOTE: translated from Chinese
            if name in mapped_completed:
                continue
            
            # NOTE: translated from Chinese (step)
            # NOTE: translated from Chinese
            
            # NOTE: translated from Chinese (step, check)
            cond = node.get('activation_condition', 'TRUE')
            if self.check_condition(cond, list(mapped_completed)):
                legal_actions.append(name)
        
        return legal_actions
    
    def suggest_robot_actions(
        self,
        current_completed_steps: Optional[List[str]],
        human_future_steps: Optional[List[str]],
    ) -> List[str]:
        """
        只负责根据 DAG + AND/OR 规则 + 预测未来动作，
        给出一组「语义上合理」的候选机器人下一步动作。
        熵的选择留给 EntropyPlanner 来做。

        修改说明：
        - 当任务含有 AND gate 时，不论当前步骤是否属于某个分支，都基于各分支的完成状态与预测未来动作来决定机器人应该进入的分支。
        - 对于每个 AND gate，只关心该 gate 对应的父节点集合及其分支叶子集合。为了避免将公共前置动作作为分支剩余任务，_build_and_gates 会移除公共叶节点。
        - 当预测未来序列为空时，若存在 "Terminate" 节点，则直接返回 Terminate；否则返回 "Wait / None"。仅在存在预测未来动作时才会考虑进入其他分支。
        - 分支选择策略：优先选择在预测未来序列中出现得最早的分支（earliest_future_pos 最小）。如果没有任何分支的叶子出现在未来序列中，再按 human_count / remaining 数量比最小选取。
        - 在分支内部，优先选择仍然未完成且出现在未来序列中的叶节点中的最早一个；如果没有，则选择该分支中编号最小的叶节点。
        """
        # NOTE: translated from Chinese (step, predict)
        completed = self._canonicalize_list(current_completed_steps)
        future = self._canonicalize_list(human_future_steps)
        future_set = set(future)
        
        # NOTE: translated from Chinese (step)
        current_step_name: Optional[str] = completed[-1] if completed else None
        current_step_id: Optional[str] = (
            self.name2id.get(current_step_name)
            if current_step_name and current_step_name in self.name2id
            else None
        )
        
        # NOTE: translated from Chinese (predict)
        if not future:
            # NOTE: translated from Chinese
            for nid, node in self.id2node.items():
                if node.get("name") == "Terminate":
                    return ["Terminate"]
            return ["Wait / None"]

        def _is_actionable(step_name: str) -> bool:
            node = self.name2node.get(step_name)
            if not node:
                return False
            # NOTE: translated from Chinese
            if node.get("midlevel_type") == "parallel":
                return False
            if _as_bool(node.get("is_midlevel", False)):
                return False
            # NOTE: translated from Chinese
            if isinstance(step_name, str) and step_name.strip().lower().startswith("start "):
                return False
            return True

        # NOTE: translated from Chinese (task)
        # NOTE: translated from Chinese
        # NOTE: translated from Chinese
        fallback_choice: Optional[str] = None
        if not self.and_gates:
            for step in future:
                if step != current_step_name and _is_actionable(step):
                    fallback_choice = step
                    break
            if fallback_choice is None:
                for step in future:
                    if _is_actionable(step):
                        fallback_choice = step
                        break
            if not (self.enable_or_fallback and self.or_gates):
                return [fallback_choice] if fallback_choice else ["Wait / None"]

        # NOTE: translated from Chinese (step)
        gate_id, current_branch_pid = self._find_and_gate_for_step(current_step_id)
        candidate_actions: List[str] = []

        if gate_id is not None:
            # NOTE: translated from Chinese
            gate = self.and_gates[gate_id]
            branch_leafs: Dict[str, Set[str]] = gate["branch_leafs"]
            parent_ids: List[str] = [str(x) for x in gate["parent_ids"]]

            branch_info: Dict[str, Dict[str, Any]] = {}
            for pid in parent_ids:
                leaf_ids = branch_leafs.get(pid, set())
                leaf_names = [self.id2node.get(i, {}).get("name", "") for i in leaf_ids if i in self.id2node]
                leaf_names = [n for n in leaf_names if n and _is_actionable(n)]
                completed_count = sum(1 for n in leaf_names if n in completed)
                remaining_names = [n for n in leaf_names if n not in completed]
                human_count = sum(1 for n in leaf_names if n in future_set)
                future_positions = [future.index(n) for n in leaf_names if n in future_set]
                earliest_pos = min(future_positions) if future_positions else None
                branch_info[pid] = {
                    "leaf_names": leaf_names,
                    "completed_count": completed_count,
                    "remaining_names": remaining_names,
                    "human_count": human_count,
                    "earliest_future_pos": earliest_pos,
                }

            candidate_pids = [pid for pid, info in branch_info.items() if info["remaining_names"]]
            if candidate_pids:
                candidate_pids = [pid for pid in candidate_pids if pid != current_branch_pid] or candidate_pids
                zero_human_pids = [pid for pid in candidate_pids if branch_info[pid]["human_count"] == 0]
                if zero_human_pids:
                    candidate_pids = zero_human_pids
                else:
                    ratios: Dict[str, float] = {}
                    for pid in candidate_pids:
                        info = branch_info[pid]
                        denom = max(len(info["remaining_names"]), 1)
                        ratios[pid] = info["human_count"] / denom
                    min_ratio = min(ratios.values())
                    candidate_pids = [pid for pid in candidate_pids if abs(ratios[pid] - min_ratio) < 1e-6]

                for pid in candidate_pids:
                    info = branch_info[pid]
                    remaining_names = info["remaining_names"]
                    if not remaining_names:
                        continue
                    rem_in_future = [n for n in future if n in remaining_names]
                    if rem_in_future:
                        chosen = rem_in_future[0]
                    else:
                        remaining_ids = [self.name2id.get(n) for n in remaining_names if n in self.name2id]
                        if remaining_ids:
                            chosen_id = min(remaining_ids, key=lambda x: int(x) if x is not None else float('inf'))
                            chosen = self.id2node.get(chosen_id, {}).get("name", "")
                            if not chosen:
                                chosen = remaining_names[0]
                        else:
                            chosen = remaining_names[0]
                    if chosen:
                        candidate_actions.append(chosen)

            # NOTE: translated from Chinese
            if candidate_actions:
                return candidate_actions

        # NOTE: translated from Chinese (step)
        # NOTE: translated from Chinese
        if gate_id is None and current_step_id is not None:
            for gid, gate in self.and_gates.items():
                parent_ids: List[str] = [str(x) for x in gate["parent_ids"]]
                # NOTE: translated from Chinese (step)
                is_on_parent_chain = False
                for pid in parent_ids:
                    if current_step_id == pid:
                        is_on_parent_chain = True
                        break
                    anc = self._get_ancestors(pid)
                    if current_step_id in anc:
                        is_on_parent_chain = True
                        break
                if not is_on_parent_chain:
                    continue

                branch_leafs: Dict[str, Set[str]] = gate["branch_leafs"]
                branch_info: Dict[str, Dict[str, Any]] = {}
                for pid in parent_ids:
                    leaf_ids = branch_leafs.get(pid, set())
                    leaf_names = [self.id2node.get(i, {}).get("name", "") for i in leaf_ids if i in self.id2node]
                    leaf_names = [n for n in leaf_names if n and _is_actionable(n)]
                    # NOTE: translated from Chinese
                    if not leaf_names:
                        continue
                    remaining_names = [n for n in leaf_names if n not in completed]
                    future_positions = [future.index(n) for n in leaf_names if n in future_set]
                    earliest_pos = min(future_positions) if future_positions else None
                    branch_info[pid] = {
                        "leaf_names": leaf_names,
                        "remaining_names": remaining_names,
                        "earliest_future_pos": earliest_pos,
                    }

                candidate_pids = [pid for pid, info in branch_info.items() if info["remaining_names"]]
                if candidate_pids:
                    # NOTE: translated from Chinese
                    candidate_pids = [pid for pid in candidate_pids if pid != current_branch_pid] or candidate_pids
                    def pid_score(pid: str) -> Any:
                        pos = branch_info[pid]["earliest_future_pos"]
                        return (pos if pos is not None else float('inf'), int(pid) if pid.isdigit() else float('inf'))
                    candidate_pids.sort(key=pid_score)
                    top_pid = candidate_pids[0]
                    remaining_names = branch_info[top_pid]["remaining_names"]
                    if remaining_names:
                        rem_in_future = [n for n in future if n in remaining_names]
                        if rem_in_future:
                            return [rem_in_future[0]]
                        remaining_ids = [self.name2id.get(n) for n in remaining_names if n in self.name2id]
                        if remaining_ids:
                            chosen_id = min(remaining_ids, key=lambda x: int(x) if x is not None else float('inf'))
                            chosen = self.id2node.get(chosen_id, {}).get("name", "")
                            if chosen:
                                return [chosen]
                        return [remaining_names[0]]

        # NOTE: translated from Chinese
        if self.enable_or_fallback and self.or_gates:
            or_gate_id, or_branch_pid = self._find_or_gate_for_step(current_step_id)
            if or_gate_id is not None:
                gate = self.or_gates[or_gate_id]
                branch_leafs: Dict[str, Set[str]] = gate["branch_leafs"]
                parent_ids: List[str] = [str(x) for x in gate["parent_ids"]]

                branch_info: Dict[str, Dict[str, Any]] = {}
                for pid in parent_ids:
                    leaf_ids = branch_leafs.get(pid, set())
                    leaf_names = [self.id2node.get(i, {}).get("name", "") for i in leaf_ids if i in self.id2node]
                    leaf_names = [n for n in leaf_names if n and _is_actionable(n)]
                    remaining_names = [n for n in leaf_names if n not in completed]
                    future_positions = [future.index(n) for n in leaf_names if n in future_set]
                    earliest_pos = min(future_positions) if future_positions else None
                    branch_info[pid] = {
                        "leaf_names": leaf_names,
                        "remaining_names": remaining_names,
                        "earliest_future_pos": earliest_pos,
                    }

                candidate_pids = [pid for pid, info in branch_info.items() if info["remaining_names"]]
                if candidate_pids:
                    # NOTE: translated from Chinese
                    def pid_score(pid: str) -> Any:
                        pos = branch_info[pid]["earliest_future_pos"]
                        return (pos if pos is not None else float('inf'), int(pid) if pid.isdigit() else float('inf'))

                    candidate_pids.sort(key=pid_score)
                    top_pid = candidate_pids[0]
                    remaining_names = branch_info[top_pid]["remaining_names"]
                    if remaining_names:
                        rem_in_future = [n for n in future if n in remaining_names]
                        if rem_in_future:
                            return [rem_in_future[0]]
                        remaining_ids = [self.name2id.get(n) for n in remaining_names if n in self.name2id]
                        if remaining_ids:
                            chosen_id = min(remaining_ids, key=lambda x: int(x) if x is not None else float('inf'))
                            chosen = self.id2node.get(chosen_id, {}).get("name", "")
                            if chosen:
                                return [chosen]
                        return [remaining_names[0]]

        # NOTE: translated from Chinese (predict)
        if "fallback_choice" in locals() and fallback_choice:
            return [fallback_choice]
        for step in future:
            if step != current_step_name and _is_actionable(step):
                return [step]
        for step in future:
            if _is_actionable(step):
                return [step]
        return ["Wait / None"]


class EntropyPlanner:
    """
    基于线程熵最小化的决策规划器
    """
    
    def __init__(self, graph_manager: TaskGraphManager):
        """
        Args:
            graph_manager: TaskGraphManager 实例
        """
        self.graph = graph_manager
    
    def calculate_entropy(
        self,
        robot_history: List[str],
        candidate_action: str,
        human_history: Optional[List[str]] = None,
    ) -> float:
        """
        计算“线程内人机混合熵”：
        - 对每个线程 k 统计人类动作数 h_k、机器人动作数 r_k
        - p_k = h_k / (h_k + r_k)，若该线程总数为 0 则跳过
        - 线程熵 H_k = -p_k log2 p_k - (1 - p_k) log2 (1 - p_k)
        - 总熵 = 按该线程动作占比 w_k = (h_k + r_k) / sum_k(h_k + r_k) 加权的 ∑ w_k H_k
        
        Args:
            robot_history: 机器人历史动作序列
            candidate_action: 候选动作
            human_history: 人类（或世界已完成）动作序列；若为 None 则只用机器人序列
            
        Returns:
            float: 线程熵值
        """
        def _is_real_action_node(name: str) -> bool:
            s = str(name or "").strip()
            if not s or s == "Wait / None":
                return False
            if s.strip().lower() == "terminate":
                return False
            node = self.graph.name2node.get(s)
            if not node:
                return False
            if _as_bool(node.get("is_midlevel", False)):
                return False
            return True

        # NOTE: translated from Chinese
        human_history = [a for a in (human_history or []) if _is_real_action_node(a)]
        robot_history = [a for a in (robot_history or []) if _is_real_action_node(a)]
        candidate_action = candidate_action if _is_real_action_node(candidate_action) else ""
        hypothetical_history = list(human_history) + list(robot_history) + ([candidate_action] if candidate_action else [])
        if not hypothetical_history:
            return 0.0
        
        # NOTE: translated from Chinese (stats)
        human_counts: Dict[str, int] = defaultdict(int)
        robot_counts: Dict[str, int] = defaultdict(int)
        for act in human_history:
            t_id = self.graph.thread_map.get(act)
            if t_id:
                human_counts[t_id] += 1
        for act in list(robot_history) + ([candidate_action] if candidate_action else []):
            t_id = self.graph.thread_map.get(act)
            if t_id:
                robot_counts[t_id] += 1
        
        # NOTE: translated from Chinese
        thread_totals: Dict[str, int] = {}
        total_actions = 0
        for tid in set(list(human_counts.keys()) + list(robot_counts.keys())):
            h = human_counts.get(tid, 0)
            r = robot_counts.get(tid, 0)
            tot = h + r
            if tot <= 0:
                continue
            thread_totals[tid] = tot
            total_actions += tot
        
        if total_actions == 0:
            return 0.0
        
        entropy = 0.0
        for tid, tot in thread_totals.items():
            h = human_counts.get(tid, 0)
            r = robot_counts.get(tid, 0)
            p = h / tot
            # NOTE: translated from Chinese
            h_bin = 0.0
            if 0 < p < 1:
                h_bin = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
            w = tot / total_actions
            entropy += w * h_bin
        
        return entropy
    
    def decide(
        self,
        current_completed_steps: Optional[List[str]],
        human_future_steps: Optional[List[str]],
        robot_history: Optional[List[str]] = None,
        human_history: Optional[List[str]] = None,
    ) -> Tuple[str, float]:
        """
        先用 TaskGraphManager.suggest_robot_actions 结合 AND 规则 + 预测未来
        得到候选动作，再用线程熵做最终选择。
        """
        if robot_history is None:
            robot_history = []
        
        # NOTE: translated from Chinese
        # NOTE: translated from Chinese
        # NOTE: translated from Chinese (valid)
        completed_for_legal = list(current_completed_steps or [])
        try:
            candidate_actions = self.graph.get_legal_robot_actions(completed_for_legal, [])
        except Exception:
            candidate_actions = []
        if not candidate_actions:
            candidate_actions = self.graph.suggest_robot_actions(current_completed_steps, human_future_steps)
        
        if not candidate_actions:
            return "Wait / None", 0.0
        
        # NOTE: translated from Chinese (filter)
        future = list(human_future_steps or [])
        future_set = set(future)

        # NOTE: translated from Chinese
        filtered: List[str] = []
        for a in candidate_actions:
            if not a or a == "Wait / None":
                continue
            node = self.graph.name2node.get(a)
            if not node:
                continue
            if node.get("midlevel_type") == "parallel":
                continue
            if _as_bool(node.get("is_midlevel", False)):
                continue
            if isinstance(a, str) and a.strip().lower().startswith("start "):
                continue
            # NOTE: translated from Chinese
            # NOTE: translated from Chinese
            if a not in future_set and not _as_bool(node.get("is_leafnode", False)):
                continue
            filtered.append(a)
        candidate_actions = filtered
        if not candidate_actions:
            return "Wait / None", 0.0

        # NOTE: translated from Chinese
        # NOTE: translated from Chinese
        # NOTE: translated from Chinese
        # NOTE: translated from Chinese
        # NOTE: translated from Chinese
        human_next = future[0] if future else None
        head_thread = self.graph.thread_map.get(human_next) if isinstance(human_next, str) else None

        # NOTE: translated from Chinese
        hh = human_history if human_history is not None else (current_completed_steps or [])

        # NOTE: translated from Chinese (stats)
        human_counts: Dict[str, int] = defaultdict(int)
        robot_counts: Dict[str, int] = defaultdict(int)
        for act in hh:
            tid = self.graph.thread_map.get(act)
            if tid:
                human_counts[tid] += 1
        for act in list(robot_history or []):
            tid = self.graph.thread_map.get(act)
            if tid:
                robot_counts[tid] += 1

        can_head_now = False
        if human_next:
            try:
                node = self.graph.name2node.get(human_next)
                cond = node.get("activation_condition", "TRUE") if node else "TRUE"
                can_head_now = self.graph.check_condition(cond, completed_for_legal)
            except Exception:
                can_head_now = False

        future_legal: List[str] = []
        unlock_legal: List[str] = []
        for action in candidate_actions:
            if not action or action == "Wait / None":
                continue
            # NOTE: translated from Chinese (filter)
            if str(action).strip().lower() == "terminate" and any(str(s).strip().lower() != "terminate" for s in future):
                continue

            if action in future_set:
                future_legal.append(action)
                continue

            if human_next and (not can_head_now):
                try:
                    node = self.graph.name2node.get(human_next)
                    cond = node.get("activation_condition", "TRUE") if node else "TRUE"
                    if self.graph.check_condition(cond, completed_for_legal + [action]):
                        unlock_legal.append(action)
                        continue
                except Exception:
                    pass

        def pick_min_entropy(actions: List[str], use_pos: bool = True) -> Tuple[str, float]:
            best_a = actions[0]
            best_e = float("inf")
            best_score = float("inf")
            for a in actions:
                e = float(self.calculate_entropy(robot_history=robot_history, candidate_action=a, human_history=hh))
                # NOTE: translated from Chinese
                pos = 10**9
                try:
                    pos = future.index(a)
                except Exception:
                    pass
                # NOTE: translated from Chinese
                POS_W = 0.05 if use_pos else 0.0
                score = e + POS_W * pos
                if score < best_score - 1e-9:
                    best_score = score
                    best_e = e
                    best_a = a
            return best_a, best_e

        # NOTE: translated from Chinese (valid)
        # NOTE: translated from Chinese
        untouched_thread_actions: List[str] = []
        for a in candidate_actions:
            tid = self.graph.thread_map.get(a)
            if not tid:
                continue
            if human_counts.get(tid, 0) == 0:
                untouched_thread_actions.append(a)
        if untouched_thread_actions:
            # NOTE: translated from Chinese
            return pick_min_entropy(untouched_thread_actions, use_pos=False)

        if future_legal:
            # NOTE: translated from Chinese
            return pick_min_entropy(future_legal, use_pos=True)
        if unlock_legal:
            return pick_min_entropy(unlock_legal, use_pos=True)

        return "Wait / None", 0.0

