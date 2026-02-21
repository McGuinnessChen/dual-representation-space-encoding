import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from eval import get_run_metrics, baseline_names, get_model_from_run
from models import build_model

sns.set_theme("notebook", "whitegrid")
palette = sns.color_palette("dark")
print(palette)

relevant_model_names = {
    "combination_regression": [
        "CoQE",
        "Transformer",
        # "Least Squares",
        # "3-Nearest Neighbors",
        # "Averaging",
    ],
    "linear_regression": [
        "CoQE",
        "Transformer",
        "Least Squares",
        # "3-Nearest Neighbors",
        # "Averaging",
    ],
    "sparse_linear_regression": [
        "CoQE",
        "Transformer",
        "Least Squares",
        # "3-Nearest Neighbors",
        # "Averaging",
        "Lasso (alpha=0.01)",
    ],
    "decision_tree": [
        "Transformer",
        "3-Nearest Neighbors",
        "2-layer NN, GD",
        "Greedy Tree Learning",
        "XGBoost",
    ],
    "relu_2nn_regression": [
        "CoQE",
        "Transformer",
        # "Least Squares",
        # "3-Nearest Neighbors",
        "2-layer NN, GD",
    ],
}

def basic_plot(metrics, models=None, trivial=1.0):
    fig, ax = plt.subplots(1, 1)

    if models is not None:
        metrics = {k: metrics[k] for k in models}

    color_id = 0
    # ax.axhline(trivial, ls="--", color="gray")
    # ax.axvline(1, ls="--", color="gray")
    start = 5
    for name, vs in metrics.items():
        x = np.arange(start, len(vs["mean"]))
        if name == "CoQE":
            color_id = 0
        elif name == "Transformer":
            color_id = 1
        elif name == "Least Squares" or name == "2-layer NN, GD":
            color_id = 2
        else:
            color_id = 4
        ax.plot(x, vs["mean"][start:], "-", label=name, color=palette[color_id % 10], lw=2)
        low = vs["bootstrap_low"][start:]
        high = vs["bootstrap_high"][start:]
        ax.fill_between(x, low, high, alpha=0.3, color=palette[color_id % 10], edgecolor="none")
        # color_id += 1
                
    ax.set_xlabel("# In-context examples", fontsize=16)
    ax.set_ylabel("MSE", fontsize=16)
    ax.set_xlim(start, len(low) + start)
    # ax.set_ylim(top=6)
    ax.tick_params(axis='x', labelsize=14)  # 修改 x 轴刻度数字大小
    ax.tick_params(axis='y', labelsize=14)  # 修改 y 轴刻度数字大小
    
    # legend = ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=14)
    # legend = ax.legend(loc="best", bbox_to_anchor=(1, 1), fontsize=14)
    # legend = ax.legend(loc="center", bbox_to_anchor=(0.6, 0.8), fontsize=14)
    fig.set_size_inches(4, 4)
    # for line in legend.get_lines():
    #     line.set_linewidth(3)

    return fig, ax

def collect_results(run_dir, df, valid_row=None, rename_eval=None, rename_model=None):
    all_metrics = {}
    for _, r in df.iterrows():
        if valid_row is not None and not valid_row(r):
            continue

        run_path = os.path.join(run_dir, r.task, r.run_id)
        _, conf = get_model_from_run(run_path, only_conf=True)

        print(r.run_name, r.run_id)
        metrics = get_run_metrics(run_path, skip_model_load=True)

        for eval_name, results in sorted(metrics.items()):
            processed_results = {}
            for model_name, m in results.items():
                if "coqe_gpt2" in model_name:
                    model_name = "CoQE"
                    if rename_model is not None:
                        model_name = rename_model(model_name, r)
                elif "gpt2" in model_name:
                    model_name = baseline_names(model_name)
                else:
                    model_name = baseline_names(model_name)
                m_processed = {}
                n_dims = conf.model.n_dims

                xlim = 2 * n_dims + 1
                # xlim = 11
                if r.task in ["relu_2nn_regression", "decision_tree", "combination_regression"]:
                    xlim = 200

                normalization = n_dims
                if r.task == "sparse_linear_regression":
                    normalization = int(r.kwargs.split("=")[-1])
                if r.task == "decision_tree":
                    normalization = 1

                for k, v in m.items():
                    v = v[:xlim]
                    # v = [vv / normalization for vv in v]
                    m_processed[k] = v
                processed_results[model_name] = m_processed
            if rename_eval is not None:
                eval_name = rename_eval(eval_name, r)
            if eval_name not in all_metrics:
                all_metrics[eval_name] = {}
            all_metrics[eval_name].update(processed_results)
    return all_metrics
