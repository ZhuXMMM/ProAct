import os
import re
import json
from collections import defaultdict
from typing import Dict, Tuple, List


def _norm_label(val: str) -> str:
    return (val or "").strip().lower()


def _compute_macro_acc_f1(stats_dict: Dict[str, Dict[str, int]]) -> Tuple[float, float]:
    """
    计算宏平均 Accuracy 与 F1：
    - 对每个类别：acc = tp / (tp + fp + fn)
                  prec = tp / (tp + fp), rec = tp / (tp + fn)
                  f1 = 2pr/(p+r)
    - 然后在所有类别上求平均
    """
    accs = []
    f1s = []
    for cls_name, counts in stats_dict.items():
        if cls_name is None:
            continue
        tp_cls = counts.get("tp", 0)
        fp_cls = counts.get("fp", 0)
        fn_cls = counts.get("fn", 0)
        denom_acc = tp_cls + fp_cls + fn_cls
        acc_cls = tp_cls / denom_acc if denom_acc > 0 else 0.0
        prec_cls = tp_cls / (tp_cls + fp_cls) if (tp_cls + fp_cls) > 0 else 0.0
        rec_cls = tp_cls / (tp_cls + fn_cls) if (tp_cls + fn_cls) > 0 else 0.0
        f1_cls = 2 * prec_cls * rec_cls / (prec_cls + rec_cls) if (prec_cls + rec_cls) > 0 else 0.0
        accs.append(acc_cls)
        f1s.append(f1_cls)
    macro_acc = sum(accs) / len(accs) if accs else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return macro_acc, macro_f1


def _levenshtein(a: List[str], b: List[str]) -> int:
    """简单的编辑距离，用于 future steps 评估。"""
    a_norm = ["".join(str(x or "").lower().split()) for x in a if str(x or "").strip()]
    b_norm = ["".join(str(x or "").lower().split()) for x in b if str(x or "").strip()]
    m, n = len(a_norm), len(b_norm)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a_norm[i - 1] == b_norm[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # delete
                dp[i][j - 1] + 1,       # insert
                dp[i - 1][j - 1] + cost # replace
            )
    return dp[m][n]


def merge_global_metrics(run_dir: str, predict_steps: int = 0) -> None:
    """
    从 run_dir/eval_pred 下的 epoch_*_rank*.jsonl 合并得到每个 epoch 的全局指标，
    并写入 run_dir/metrics.csv（覆盖原有文件）。
    """
    eval_dir = os.path.join(run_dir, "eval_pred")
    if not os.path.isdir(eval_dir):
        print(f"[merge_global_metrics] eval_pred 目录不存在: {eval_dir}")
        os.makedirs(eval_dir, exist_ok=True)
        print(f"[merge_global_metrics] eval_pred 目录创建成功: {eval_dir}")
        # return

    # NOTE: translated from Chinese
    files = [f for f in os.listdir(eval_dir) if "rank" in f and f.endswith(".jsonl")]
    if not files:
        print(f"[merge_global_metrics] 未找到任何预测文件，跳过全局指标合并")
        return

    # NOTE: translated from Chinese
    # - epoch_1.rank0.jsonl / epoch_1_rank0.jsonl
    # - 10.rank0.jsonl / 10_rank0.jsonl
    pattern = re.compile(r"(?:epoch_)?(?P<epoch>[^._]+)[._]rank\d+\.jsonl$")
    eval_only_pattern = re.compile(r"eval_only[._]rank\d+\.jsonl$")
    epoch_to_files: Dict[str, list] = defaultdict(list)
    for name in files:
        m = pattern.match(name)
        if m:
            ep = m.group("epoch")
            epoch_to_files[ep].append(os.path.join(eval_dir, name))
            continue
        if eval_only_pattern.match(name):
            epoch_to_files["eval_only"].append(os.path.join(eval_dir, name))

    if not epoch_to_files:
        print(f"[merge_global_metrics] 未匹配到 epoch_*_rank*.jsonl，跳过")
        return

    # NOTE: translated from Chinese
    train_val_map: Dict[str, Tuple[str, str]] = {}
    orig_metrics = os.path.join(run_dir, "metrics.csv")
    if os.path.exists(orig_metrics):
        try:
            with open(orig_metrics, "r", encoding="utf-8") as f:
                header = f.readline().strip().split(",")
                idx_epoch = header.index("epoch") if "epoch" in header else None
                idx_train = header.index("train_loss") if "train_loss" in header else None
                idx_val = header.index("val_loss") if "val_loss" in header else None
                for line in f:
                    if not line.strip():
                        continue
                    parts = line.strip().split(",")
                    if idx_epoch is None or idx_train is None or idx_val is None:
                        continue
                    ep = parts[idx_epoch]
                    train_loss = parts[idx_train]
                    val_loss = parts[idx_val]
                    train_val_map[str(ep)] = (train_loss, val_loss)
        except Exception:
            pass

    # NOTE: translated from Chinese (step)
    # NOTE: translated from Chinese
    future_any = predict_steps > 0
    if not future_any:
        for ep_files in epoch_to_files.values():
            if future_any:
                break
            for path in ep_files:
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                rec = json.loads(line)
                            except Exception:
                                continue
                            pred_future = rec.get("pred_future_steps") or []
                            gt_future = rec.get("gt_future_steps") or []
                            if (isinstance(pred_future, list) and len(pred_future) > 0) or (isinstance(gt_future, list) and len(gt_future) > 0):
                                future_any = True
                                break
                        if future_any:
                            break
                except Exception:
                    continue

    out_csv = os.path.join(run_dir, "metrics.csv")
    # NOTE: translated from Chinese (check)
    try:
        expected_world = int(os.environ.get("WORLD_SIZE", "0"))
    except Exception:
        expected_world = 0

    if expected_world > 0:
        for ep, ep_files in epoch_to_files.items():
            if len(ep_files) < expected_world:
                print(f"[merge_global_metrics] epoch {ep}: 发现 {len(ep_files)}/{expected_world} 个 rank 文件，请检查是否有 rank 未写出 eval_pred。")
    header_cols = [
        "epoch","train_loss","val_loss",
        "trig_mAcc","trig_mF1","trig_Acc","trig_F1",
        "task_mAcc","task_mF1","task_Acc","task_F1",
        "step_mAcc","step_mF1","step_Acc","step_F1",
        "avg_generation_time",
    ]
    if future_any:
        header_cols.append("future_edit_dist")

    with open(out_csv, "w", encoding="utf-8") as wf:
        wf.write(",".join(header_cols) + "\n")

        # NOTE: translated from Chinese
        for ep in sorted(epoch_to_files, key=lambda x: (str(x))):
            ep_files = epoch_to_files[ep]
            tp = fp = tn = fn = 0
            task_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
            step_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
            total_gen_time = 0.0
            gen_count = 0
            future_edit_sum = 0.0
            future_edit_count = 0
            future_enabled = False

            for path in ep_files:
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            rec = json.loads(line)
                            pred_is = int(rec.get("pred", 0))
                            gt = int(rec.get("gt", 0))
                            pred_task = _norm_label(rec.get("pred_task_matched") or rec.get("pred_task") or "")
                            gt_task = _norm_label(rec.get("gt_task") or "")
                            pred_step = _norm_label(rec.get("pred_step_matched") or rec.get("pred_step") or "")
                            gt_step = _norm_label(rec.get("gt_step") or "")
                            gen_t = float(rec.get("generation_time", 0.0))
                            pred_future = rec.get("pred_future_steps") or []
                            gt_future = rec.get("gt_future_steps") or []
                            if pred_future or gt_future:
                                future_enabled = True
                                try:
                                    future_edit_sum += _levenshtein(pred_future if isinstance(pred_future, list) else [], gt_future if isinstance(gt_future, list) else [])
                                    future_edit_count += 1
                                except Exception:
                                    pass

                            total_gen_time += gen_t
                            gen_count += 1

                            if pred_is == 1 and gt == 1:
                                tp += 1
                            elif pred_is == 1 and gt == 0:
                                fp += 1
                            elif pred_is == 0 and gt == 0:
                                tn += 1
                            else:
                                fn += 1

                            # NOTE: translated from Chinese
                            if gt == 1:
                                if pred_is == 1:
                                    if gt_task:
                                        if pred_task == gt_task:
                                            task_stats[gt_task]["tp"] += 1
                                        else:
                                            task_stats[gt_task]["fn"] += 1
                                            if pred_task:
                                                task_stats[pred_task]["fp"] += 1
                                    elif pred_task:
                                        task_stats[pred_task]["fp"] += 1
                                else:
                                    if gt_task:
                                        task_stats[gt_task]["fn"] += 1
                            elif pred_is == 1 and pred_task:
                                task_stats[pred_task]["fp"] += 1

                            # NOTE: translated from Chinese
                            if gt == 1:
                                if pred_is == 1:
                                    if gt_step:
                                        if pred_step == gt_step:
                                            step_stats[gt_step]["tp"] += 1
                                        else:
                                            step_stats[gt_step]["fn"] += 1
                                            if pred_step:
                                                step_stats[pred_step]["fp"] += 1
                                    elif pred_step:
                                        step_stats[pred_step]["fp"] += 1
                                else:
                                    if gt_step:
                                        step_stats[gt_step]["fn"] += 1
                            elif pred_is == 1 and pred_step:
                                step_stats[pred_step]["fp"] += 1
                except Exception as e:
                    print(f"[merge_global_metrics] 读取 {path} 失败: {e}")

            # NOTE: translated from Chinese
            pos_prec = tp / (tp + fp + 1e-9)
            pos_recall = tp / (tp + fn + 1e-9)
            val_acc = (tp + tn) / max(1, (tp + tn + fp + fn))
            trig_f1_micro = 2 * (pos_prec * pos_recall) / (pos_prec + pos_recall + 1e-9)

            # NOTE: translated from Chinese
            trig_stats = {
                "pos": {"tp": tp, "fp": fp, "fn": fn},
                "neg": {"tp": tn, "fp": fn, "fn": fp},
            }
            trig_macc, trig_mf1 = _compute_macro_acc_f1(trig_stats)

            # NOTE: translated from Chinese
            task_tp = sum(v["tp"] for v in task_stats.values())
            task_fp = sum(v["fp"] for v in task_stats.values())
            task_fn = sum(v["fn"] for v in task_stats.values())
            task_acc = task_tp / max(1, (task_tp + task_fp + task_fn))
            task_prec = task_tp / max(1, (task_tp + task_fp))
            task_rec = task_tp / max(1, (task_tp + task_fn))
            task_f1 = 2 * (task_prec * task_rec) / max(1e-9, (task_prec + task_rec))
            task_macc, task_mf1 = _compute_macro_acc_f1(task_stats)

            step_tp = sum(v["tp"] for v in step_stats.values())
            step_fp = sum(v["fp"] for v in step_stats.values())
            step_fn = sum(v["fn"] for v in step_stats.values())
            step_acc = step_tp / max(1, (step_tp + step_fp + step_fn))
            step_prec = step_tp / max(1, (step_tp + step_fp))
            step_rec = step_tp / max(1, (step_tp + step_fn))
            step_f1 = 2 * (step_prec * step_rec) / max(1e-9, (step_prec + step_rec))
            step_macc, step_mf1 = _compute_macro_acc_f1(step_stats)
            avg_generation_time = total_gen_time / max(1, gen_count)
            future_edit_avg = future_edit_sum / max(1, future_edit_count) if future_enabled else None

            train_loss, val_loss = train_val_map.get(str(ep), ("", ""))
            # NOTE: translated from Chinese
            train_loss_str = str(train_loss) if train_loss not in (None, "") else ""
            val_loss_str = str(val_loss) if val_loss not in (None, "") else ""
            fields = [
                ep,
                train_loss_str,
                val_loss_str,
                f"{trig_macc:.6f}",
                f"{trig_mf1:.6f}",
                f"{val_acc:.6f}",
                f"{trig_f1_micro:.6f}",
                f"{task_macc:.6f}",
                f"{task_mf1:.6f}",
                f"{task_acc:.6f}",
                f"{task_f1:.6f}",
                f"{step_macc:.6f}",
                f"{step_mf1:.6f}",
                f"{step_acc:.6f}",
                f"{step_f1:.6f}",
                f"{avg_generation_time:.6f}",
            ]
            if future_any:
                fields.append(f"{future_edit_avg:.6f}" if future_enabled and future_edit_avg is not None else "")
            wf.write(",".join(fields) + "\n")

    print(f"[merge_global_metrics] 全局指标已写入: {out_csv}")


