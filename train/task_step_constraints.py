"""
Task-Step 约束模块：从 annotation 中提取 task → steps 映射，用于约束生成和增强训练
"""
import json
from typing import Dict, List, Set, Optional
from pathlib import Path
import difflib
import torch


class TaskStepMapper:
    """
    从 annotation.json 中提取 task → steps 的层级关系
    用于约束 step 生成，确保 step 属于对应的 task
    """
    
    def __init__(self, annotation_path: str):
        """
        Args:
            annotation_path: all_annotations.json 的路径
        """
        self.annotation_path = annotation_path
        self.task_to_steps: Dict[str, List[str]] = {}  # task_name -> [step_names]
        self.step_to_task: Dict[str, str] = {}  # step_name -> task_name
        self.all_tasks: Set[str] = set()
        self.all_steps: Set[str] = set()
        
        self._load_mapping()
    
    def _load_mapping(self):
        """从 annotation 文件中加载 task-step 映射"""
        try:
            with open(self.annotation_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"警告：无法加载 annotation 文件 {self.annotation_path}: {e}")
            return
        
        # NOTE: translated from Chinese
        annotations = data.get('annotations', {})
        
        for video_id, ann in annotations.items():
            # NOTE: translated from Chinese (task)
            task_name = ann.get('task_name_en') or ann.get('task_name') or ''
            if not task_name or not isinstance(task_name, str):
                continue
            
            task_name = task_name.strip()
            self.all_tasks.add(task_name)
            
            # NOTE: translated from Chinese
            segments = ann.get('segments', [])
            if not isinstance(segments, list):
                continue
            
            # NOTE: translated from Chinese
            step_names = []
            for segment in segments:
                if not isinstance(segment, dict):
                    continue
                
                # NOTE: translated from Chinese
                step_name = segment.get('step_name', '')
                if not step_name:
                    continue
                
                step_name = step_name.strip()
                if step_name:
                    step_names.append(step_name)
                    self.all_steps.add(step_name)
                    self.step_to_task[step_name] = task_name
            
            # NOTE: translated from Chinese
            if task_name not in self.task_to_steps:
                self.task_to_steps[task_name] = []
            
            for step_name in step_names:
                if step_name not in self.task_to_steps[task_name]:
                    self.task_to_steps[task_name].append(step_name)
        
        print(f"Task-Step 映射加载完成：")
        print(f"  - 任务数: {len(self.task_to_steps)}")
        print(f"  - 步骤数（去重）: {len(self.all_steps)}")
        print(f"  - 平均每个任务的步骤数: {sum(len(v) for v in self.task_to_steps.values()) / max(1, len(self.task_to_steps)):.1f}")
    
    def get_valid_steps(self, task_name: str, fuzzy_match: bool = True) -> List[str]:
        """
        获取给定任务下的合法步骤列表
        
        Args:
            task_name: 任务名称
            fuzzy_match: 是否使用模糊匹配（针对任务名称可能有微小差异）
        
        Returns:
            该任务下的合法步骤名称列表
        """
        if not task_name:
            return []
        
        task_name = task_name.strip()
        
        # NOTE: translated from Chinese
        if task_name in self.task_to_steps:
            return self.task_to_steps[task_name]
        
        # NOTE: translated from Chinese
        if fuzzy_match:
            matches = difflib.get_close_matches(task_name, self.task_to_steps.keys(), n=1, cutoff=0.8)
            if matches:
                matched_task = matches[0]
                print(f"模糊匹配: '{task_name}' -> '{matched_task}'")
                return self.task_to_steps[matched_task]
        
        # NOTE: translated from Chinese
        return []
    
    def is_valid_task_step_pair(self, task_name: str, step_name: str, fuzzy_match: bool = True) -> bool:
        """
        检查 task-step 组合是否合法
        
        Args:
            task_name: 任务名称
            step_name: 步骤名称
            fuzzy_match: 是否使用模糊匹配
        
        Returns:
            True 如果组合合法，False 否则
        """
        if not task_name or not step_name:
            return False
        
        valid_steps = self.get_valid_steps(task_name, fuzzy_match=fuzzy_match)
        if not valid_steps:
            return False
        
        step_name = step_name.strip()
        
        # NOTE: translated from Chinese
        if step_name in valid_steps:
            return True
        
        # NOTE: translated from Chinese
        if fuzzy_match:
            # NOTE: translated from Chinese
            step_norm = step_name.lower().replace(' ', '')
            for valid_step in valid_steps:
                valid_norm = valid_step.lower().replace(' ', '')
                if step_norm == valid_norm:
                    return True
                # NOTE: translated from Chinese
                if difflib.SequenceMatcher(None, step_norm, valid_norm).ratio() > 0.85:
                    return True
        
        return False
    
    def get_task_for_step(self, step_name: str) -> Optional[str]:
        """
        根据 step 名称获取其所属的 task
        
        Args:
            step_name: 步骤名称
        
        Returns:
            任务名称，如果未找到则返回 None
        """
        if not step_name:
            return None
        
        step_name = step_name.strip()
        return self.step_to_task.get(step_name)


class TaskStepConstraintLoss:
    """
    Task-Step 约束损失：对不合法的 task-step 组合施加额外惩罚
    """
    
    def __init__(self, mapper: TaskStepMapper, penalty_weight: float = 1.0):
        """
        Args:
            mapper: TaskStepMapper 实例
            penalty_weight: 惩罚权重
        """
        self.mapper = mapper
        self.penalty_weight = penalty_weight
    
    def compute_constraint_loss(
        self,
        pred_task: str,
        pred_step: str,
        task_logits: torch.Tensor,  # NOTE: translated from Chinese
        step_logits: torch.Tensor,  # NOTE: translated from Chinese
        is_trigger: bool,
    ) -> Optional[torch.Tensor]:
        """
        计算约束损失：如果预测的 task-step 组合不合法，返回惩罚
        
        注意：这需要在生成阶段使用，训练阶段可以用下面的方法
        """
        if not is_trigger:
            return None
        
        if not self.mapper.is_valid_task_step_pair(pred_task, pred_step):
            # NOTE: translated from Chinese (valid)
            # NOTE: translated from Chinese
            return torch.tensor(self.penalty_weight, device=task_logits.device)
        
        return None


def build_step_token_mapping(tokenizer, mapper: TaskStepMapper) -> Dict[str, List[int]]:
    """
    构建 task → valid_step_token_ids 的映射
    
    Args:
        tokenizer: HuggingFace tokenizer
        mapper: TaskStepMapper 实例
    
    Returns:
        {task_name: [step_token_id_1, step_token_id_2, ...]}
    """
    task_to_step_tokens: Dict[str, List[int]] = {}
    
    for task_name, step_names in mapper.task_to_steps.items():
        step_token_ids = []
        for step_name in step_names:
            # NOTE: translated from Chinese
            tokens = tokenizer.encode(step_name, add_special_tokens=False)
            if tokens:
                # NOTE: translated from Chinese
                step_token_ids.append(tokens[0])
        
        task_to_step_tokens[task_name] = list(set(step_token_ids))  # NOTE: translated from Chinese
    
    return task_to_step_tokens


# NOTE: translated from Chinese
if __name__ == "__main__":
    import sys
    
    # NOTE: translated from Chinese
    annotation_path = os.environ.get("ANNOTATION_JSON", "data/annotations/all_annotations.json")
    mapper = TaskStepMapper(annotation_path)
    
    # NOTE: translated from Chinese
    print("\n示例查询：")
    task = "Make coffee"
    steps = mapper.get_valid_steps(task)
    print(f"任务 '{task}' 的合法步骤：{steps}")
    
    # NOTE: translated from Chinese (check, valid)
    print(f"\n检查 task-step 组合：")
    print(f"  ('Make coffee', 'Pour water'): {mapper.is_valid_task_step_pair('Make coffee', 'Pour water')}")
    print(f"  ('Make coffee', 'Invalid step'): {mapper.is_valid_task_step_pair('Make coffee', 'Invalid step')}")

