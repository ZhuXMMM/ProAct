#!/usr/bin/env python3
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


class TaskGraphManager:
    """
    任务图管理器：负责加载 taxonomy、识别并行线程、判断动作合法性
    """
    
    def __init__(self, annotation_path: str, task_name: str):
        """
        Args:
            annotation_path: 包含 "vocabulary" 和 "taxonomy" 的大 JSON 路径
            task_name: 当前视频对应的任务名称 (e.g., "Assemble Bed")
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
        
        # NOTE: translated from Chinese (step)
        # NOTE: translated from Chinese
        self.thread_map = self._build_thread_map()
        
        # NOTE: translated from Chinese
        self.and_gates = self._build_and_gates()
    
    def _build_thread_map(self) -> Dict[str, str]:
        """
        静态分析算法：
        1. 找到所有 midlevel_type == 'parallel' 的节点。
        2. 对该节点的每一个直接子节点，启动 DFS/BFS 遍历。
        3. 将该子分支下的所有后代节点标记为同一个 Thread ID。
        """
        thread_map = {}
        visited = set()
        
        # NOTE: translated from Chinese
        parallel_parents = [nid for nid, node in self.id2node.items() 
                            if node.get('midlevel_type') == 'parallel']
        
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
            
            # NOTE: translated from Chinese
            for i, child_id in enumerate(children):
                thread_id = f"{p_id}_thread_{i}"
                # NOTE: translated from Chinese (iterate)
                stack = [child_id]
                branch_visited = set()
                
                while stack:
                    curr = stack.pop()
                    if curr in branch_visited:
                        continue
                    branch_visited.add(curr)
                    
                    # NOTE: translated from Chinese
                    if curr not in self.id2node:
                        continue
                    
                    node_name = self.id2node[curr].get('name', '')
                    if node_name:
                        # NOTE: translated from Chinese
                        if node_name not in thread_map:
                            thread_map[node_name] = thread_id
                    
                    # NOTE: translated from Chinese
                    for next_nid, next_node in self.id2node.items():
                        pids = next_node.get('parent_id')
                        if pids:
                            if not isinstance(pids, list):
                                pids = [pids]
                            if str(curr) in [str(p) for p in pids]:
                                stack.append(next_nid)
        
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
            if "AND" not in cond:
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
                    if self.id2node.get(str(x), {}).get("is_leafnode", False)
                }
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
    
    def _canonicalize_list(self, raw_steps: Optional[List[str]]) -> List[str]:
        """把一个字符串列表模糊匹配成 DAG 里的标准步骤名。"""
        if not raw_steps:
            return []
        
        all_names = list(self.name2id.keys())
        mapped: List[str] = []
        
        for s in raw_steps:
            if not isinstance(s, str):
                continue
            m = self._fuzzy_match_step_name(s, all_names)
            if m and m not in mapped:
                mapped.append(m)
        
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
        
        completed_ids = set()
        for name in completed_step_names:
            if name in self.name2id:
                completed_ids.add(str(self.name2id[name]))
        
        # NOTE: translated from Chinese
        def replace_logic(match):
            nid = match.group(1)
            return "True" if nid in completed_ids else "False"
        
        # NOTE: translated from Chinese
        # NOTE: translated from Chinese
        eval_str = re.sub(r'\((\d+)\)', replace_logic, condition_str)
        
        # NOTE: translated from Chinese
        eval_str = eval_str.replace('AND', 'and').replace('OR', 'or')
        
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
        
        排除掉：
        1. 已经做完的。
        2. 前置条件不满足的。
        3. 人类即将要做的 (human_future_steps)。
        
        Args:
            current_completed_steps: 当前已完成的步骤名称列表（支持模糊匹配）
            human_future_steps: 人类即将执行的步骤名称列表（支持模糊匹配）
            
        Returns:
            List[str]: 合法的机器人动作列表
        """
        legal_actions = []
        
        # NOTE: translated from Chinese
        all_leaf_names = [node.get('name', '') for node in self.id2node.values() 
                         if node.get('is_leafnode', False) and node.get('name', '')]
        
        # NOTE: translated from Chinese (step)
        mapped_completed = set()
        for step in current_completed_steps:
            if step in self.name2node:
                mapped_completed.add(step)
            else:
                # NOTE: translated from Chinese
                matched = self._fuzzy_match_step_name(step, all_leaf_names)
                if matched:
                    mapped_completed.add(matched)
        
        mapped_human_future = set()
        for step in human_future_steps:
            if step in self.name2node:
                mapped_human_future.add(step)
            else:
                # NOTE: translated from Chinese
                matched = self._fuzzy_match_step_name(step, all_leaf_names)
                if matched:
                    mapped_human_future.add(matched)
        
        for nid, node in self.id2node.items():
            name = node.get('name', '')
            if not name:
                continue
            
            # NOTE: translated from Chinese
            if name in mapped_completed:
                continue
            
            # NOTE: translated from Chinese
            if name in mapped_human_future:
                continue
            
            # NOTE: translated from Chinese
            if not node.get('is_leafnode', False):
                continue
            
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

        # NOTE: translated from Chinese (task, predict)
        if not self.and_gates:
            # NOTE: translated from Chinese (step, predict)
            for step in future:
                if step != current_step_name:
                    return [step]
            return [future[0]]

        # NOTE: translated from Chinese (step)
        gate_id, current_branch_pid = self._find_and_gate_for_step(current_step_id)
        # NOTE: translated from Chinese (step, predict)
        if gate_id is None:
            # NOTE: translated from Chinese
            for step in future:
                if step != current_step_name:
                    return [step]
            return [future[0]]

        # NOTE: translated from Chinese
        gate = self.and_gates[gate_id]
        branch_leafs: Dict[str, Set[str]] = gate["branch_leafs"]
        parent_ids: List[str] = [str(x) for x in gate["parent_ids"]]

        # NOTE: translated from Chinese (task, stats)
        branch_info: Dict[str, Dict[str, Any]] = {}
        for pid in parent_ids:
            leaf_ids = branch_leafs.get(pid, set())
            # NOTE: translated from Chinese
            leaf_names = [self.id2node.get(i, {}).get("name", "") for i in leaf_ids if i in self.id2node]
            leaf_names = [n for n in leaf_names if n]
            # NOTE: translated from Chinese (stats)
            completed_count = sum(1 for n in leaf_names if n in completed)
            remaining_names = [n for n in leaf_names if n not in completed]
            # NOTE: translated from Chinese
            human_count = sum(1 for n in leaf_names if n in future_set)
            # NOTE: translated from Chinese
            future_positions = [future.index(n) for n in leaf_names if n in future_set]
            earliest_pos = min(future_positions) if future_positions else None
            branch_info[pid] = {
                "leaf_names": leaf_names,
                "completed_count": completed_count,
                "remaining_names": remaining_names,
                "human_count": human_count,
                "earliest_future_pos": earliest_pos,
            }

        # NOTE: translated from Chinese (task)
        candidate_pids = [pid for pid, info in branch_info.items() if info["remaining_names"]]
        if not candidate_pids:
            # NOTE: translated from Chinese (predict)
            for step in future:
                if step != current_step_name:
                    return [step]
            return [future[0]]

        # NOTE: translated from Chinese (step)
        candidate_pids = [pid for pid in candidate_pids if pid != current_branch_pid] or candidate_pids

        # NOTE: translated from Chinese
        zero_human_pids = [pid for pid in candidate_pids if branch_info[pid]["human_count"] == 0]
        if zero_human_pids:
            candidate_pids = zero_human_pids
        else:
            # NOTE: translated from Chinese
            ratios: Dict[str, float] = {}
            for pid in candidate_pids:
                info = branch_info[pid]
                denom = max(len(info["remaining_names"]), 1)
                ratios[pid] = info["human_count"] / denom
            min_ratio = min(ratios.values())
            candidate_pids = [pid for pid in candidate_pids if abs(ratios[pid] - min_ratio) < 1e-6]

        # NOTE: translated from Chinese
        candidate_actions: List[str] = []
        for pid in candidate_pids:
            info = branch_info[pid]
            remaining_names = info["remaining_names"]
            if not remaining_names:
                continue
            # NOTE: translated from Chinese
            rem_in_future = [n for n in future if n in remaining_names]
            if rem_in_future:
                chosen = rem_in_future[0]
            else:
                # NOTE: translated from Chinese
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

        # NOTE: translated from Chinese (predict)
        if not candidate_actions:
            for step in future:
                if step != current_step_name:
                    return [step]
            return [future[0]]

        return candidate_actions


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
    
    def calculate_entropy(self, robot_history: List[str], candidate_action: str) -> float:
        """
        计算假如机器人执行 candidate_action 后，机器人行为序列的线程熵。
        
        公式：H(r) = - sum (p * log2(p))
        其中 p = N_{thread_i} / N_total
        
        Args:
            robot_history: 机器人历史动作序列
            candidate_action: 候选动作
            
        Returns:
            float: 线程熵值
        """
        # NOTE: translated from Chinese
        hypothetical_history = robot_history + [candidate_action]
        if not hypothetical_history:
            return 0.0
        
        # NOTE: translated from Chinese (stats)
        thread_counts = defaultdict(int)
        total_thread_actions = 0
        
        for action in hypothetical_history:
            t_id = self.graph.thread_map.get(action)
            if t_id:
                thread_counts[t_id] += 1
                total_thread_actions += 1
            # NOTE: translated from Chinese (step)
            # NOTE: translated from Chinese
        
        if total_thread_actions == 0:
            return 0.0
        
        # NOTE: translated from Chinese
        entropy = 0.0
        for count in thread_counts.values():
            p = count / total_thread_actions
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    def decide(
        self,
        current_completed_steps: Optional[List[str]],
        human_future_steps: Optional[List[str]],
        robot_history: Optional[List[str]] = None,
    ) -> Tuple[str, float]:
        """
        先用 TaskGraphManager.suggest_robot_actions 结合 AND 规则 + 预测未来
        得到候选动作，再用线程熵做最终选择。
        """
        if robot_history is None:
            robot_history = []
        
        # NOTE: translated from Chinese
        candidate_actions = self.graph.suggest_robot_actions(
            current_completed_steps, human_future_steps
        )
        
        if not candidate_actions:
            return "Wait / None", 0.0
        
        # NOTE: translated from Chinese
        best_action = None
        best_entropy = float("inf")
        
        for action in candidate_actions:
            entropy = self.calculate_entropy(robot_history, action)
            if entropy < best_entropy:
                best_entropy = entropy
                best_action = action
        
        # NOTE: translated from Chinese
        if best_action is None:
            best_action = candidate_actions[0]
            best_entropy = self.calculate_entropy(robot_history, best_action)
        
        return best_action, best_entropy

