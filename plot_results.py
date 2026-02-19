#!/usr/bin/env python3
"""
plot_acc.py  <log_dir>
遍历指定目录下的所有 TensorBoard events 文件，提取 accuracy_closed 并画图。
示例：
    python plot_acc.py ./eval_fewshot_holdouts
"""
import os
import re
import argparse
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

def main(log_dir, acc_mode):
    files = [
        os.path.join(log_dir, f)
        for f in os.listdir(log_dir)
        if f.startswith('events.out.tfevents')
    ]
    if not files:
        print(f'目录 {log_dir} 下没有找到 events 文件。')
        return

    steps, accs = [], []
    for fpath in files:
        ea = EventAccumulator(fpath)
        ea.Reload()
        # print(ea.Tags())
        # tensor_events = ea.Tensors('accuracy_closed')
        # if not tensor_events:
        #     continue
        # print(tensor_events)
        ev = ea.Tensors(acc_mode)[0]
        step = ev.step                 # ← 真正的 step
        val  = np.frombuffer(ev.tensor_proto.tensor_content, np.float32)[0]
        accs.append(float(val))
        steps.append(step)

    if not steps:
        print('没有找到 accuracy 数据。')
        return

    steps_sorted, accs_sorted = zip(*sorted(zip(steps, accs)))
    steps, accs = np.array(list(steps_sorted)), np.array(list(accs_sorted))

    if len(steps) == 50:
        steps = steps[::2]
        accs = (accs[::2] + accs[1::2]) / 2
    elif len(steps) != 25:
        _, unique_indices = np.unique(steps, return_index=True)
        steps = steps[unique_indices]
        accs = accs[unique_indices]

    steps = np.insert(steps,0,0)
    accs = np.insert(accs,0,0.5)
    return steps, accs



if __name__ == '__main__':

    acc_mode = "accuracy_closed"
    eval_mode = "eval_fewshot_holdout"
    out_path = f'emergent_in_context_learning/icl_{acc_mode}_0910.png'
    # steps, accs = main(f'emergent_in_context_learning/dual_transformer_omniglot_28_layer4_mu5_small/{eval_mode}', acc_mode)
    steps2, accs2 = main(f'emergent_in_context_learning/dual_transformer_omniglot_39_layer4_mu5_small_p0.9_z1_emb256/{eval_mode}', acc_mode)
    # steps3, accs3 = main(f'emergent_in_context_learning/dual_transformer_omniglot_38_layer4_mu5_small_p0.9_z2/{eval_mode}', acc_mode)
    # steps4, accs4 = main(f'emergent_in_context_learning/transformer_omniglot_4_embd64/{eval_mode}', acc_mode)
    # steps5, accs5 = main(f'emergent_in_context_learning/transformer_omniglot_6_layer12_p0.9_z1/{eval_mode}', acc_mode)
    # steps6, accs6 = main(f'emergent_in_context_learning/transformer_omniglot_9_layer12_p0.9_z2/{eval_mode}', acc_mode)
    # steps_baseline, accs_baseline = main('emergent_in_context_learning/transformer_omniglot/eval_fewshot_holdout', acc_mode)

    # print(main('emergent_in_context_learning/transformer_omniglot_4_embd64/eval_fewshot_holdout', acc_mode))


    # print("Steps", steps, "Acc", accs)
    print("Steps2", steps2, "Acc", accs2)
    # print("Steps3", steps3, "Acc", accs3)
    # print("Steps4", steps4, "Acc", accs4)
    # print("Steps5", steps5, "Acc", accs5)
    # print("Steps6", steps6, "Acc", accs6)
    # print("Steps_baseline", steps_baseline, "Acc", accs_baseline)

    # plt.figure(figsize=(6, 4))
    # plt.plot(steps, accs, marker='o', label='ours')
    # plt.plot(steps2, accs2, marker='o', label='ours2')
    # plt.plot(steps3, accs3, marker='o', label='ours3')
    # plt.plot(steps4, accs4, marker='o', label='ours4')
    # plt.plot(steps5, accs5, marker='o', label='ours5')
    # plt.plot(steps6, accs6, marker='o', label='ours6')
    # plt.plot(steps_baseline, accs_baseline, marker='o', label='baseline')
    # plt.legend()
    # plt.xlim(0, 1e5)
    # plt.xlabel('train steps')
    # plt.ylabel(acc_mode)
    # plt.title(f'ICL {acc_mode} on holdout classes')
    # plt.grid(True)
    # plt.tight_layout()
    #
    # plt.savefig(out_path, dpi=200)
    # plt.show()
    # print(f'已保存为 {out_path}')