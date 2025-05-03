import os
import numpy as np
import re
import matplotlib.pyplot as plt

# ----------- 读取预测值与真实值 -----------
def read_prediction_txt(file_path):
    pred_list = []
    label_list = []
    with open(file_path, 'r', encoding='gbk') as f:
        lines = f.readlines()
    for line in lines:
        matches = re.findall(r'\[([-\d.\s]+)\]', line)
        if len(matches) == 2:
            pred = list(map(float, matches[0].split()))
            label = list(map(float, matches[1].split()))
            pred_list.append(pred)
            label_list.append(label)
    return np.array(pred_list), np.array(label_list)

# ----------- 绘图函数 -----------
def plot_figure(save_path, label, pred, title_suffix="plot"):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 24))
    plot_single_axis(ax1, pred[:, 0], label[:, 0], 'X')
    plot_single_axis(ax2, pred[:, 1], label[:, 1], 'Y')
    plot_single_axis(ax3, pred[:, 2], label[:, 2], 'Z')
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"result_plot_{title_suffix}.png"), dpi=300, bbox_inches='tight')
    plt.show()

def plot_single_axis(ax, pred, label, axis_label):
    ax.plot(pred, color='red', linewidth=2.5, label=f'Predicted load value in $\\it{{{axis_label}}}$')
    ax.plot(label, color='black', linewidth=2.5, label=f'True load value in $\\it{{{axis_label}}}$')
    ax.set_xlabel("Data Sequence", fontsize=18, fontweight='bold')
    ax.set_ylabel("Load (N)", fontsize=18, fontweight='bold', color='black')
    ax.tick_params(axis='both', which='major', labelsize=12, width=2.0, length=6)


    epsilon = 1e-8
    relative_error = np.abs((label - pred) / (label + epsilon)) * 100
    ax2 = ax.twinx()
    ax2.plot(relative_error, 'b^', label=f'Relative Error in $\\it{{{axis_label}}}$', markersize=7,
             markerfacecoloralt='white', fillstyle='right')
    ax2.set_ylabel("Relative Error (%)", fontsize=18, fontweight='bold', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue', labelsize=12, width=2.0, length=6)

    current_ylim = ax2.get_ylim()
    if current_ylim[1] < 8:
        ax2.set_ylim(current_ylim[0], 7)
    else:
        ax2.set_ylim(0, current_ylim[1] * 1.3)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    legend = ax2.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=10)


# ----------- 主程序 -----------
if __name__ == "__main__":
    txt_file = "./result_best/test_results_work_18.txt"   # 修改为你的 .txt 文件路径
    save_path = "figures"                   # 图片保存路径
    pred, label = read_prediction_txt(txt_file)
    plot_figure(save_path, label, pred, title_suffix="18")  # "19" 用于生成文件名
