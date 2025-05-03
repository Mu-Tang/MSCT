import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from sklearn.metrics import mean_squared_error
from src.dataloader import *
import numpy as np


def mae(true_values, pred_value):
    """
    计算平均绝对误差
    :param true_values:
    :param pred_value:
    :return:
    """
    error = mean_absolute_error(true_values, pred_value)
    return error


def amae(label, pred):
    """
    得到各轴上的平均绝对误差
    :param label:
    :param pred:
    :return:
    """
    pred, label = np.array(pred), np.array(label)
    x_pred, y_pred, z_pred = pred[:, 0], pred[:, 1], pred[:, 2]
    x_label, y_label, z_label = label[:, 0], label[:, 1], label[:, 2]
    mean_absolute_error_x = mae(x_label, x_pred)
    mean_absolute_error_y = mae(y_label, y_pred)
    mean_absolute_error_z = mae(z_label, z_pred)

    return mean_absolute_error_x, mean_absolute_error_y, mean_absolute_error_z


def mape(true_value, pred_value):
    """
    计算平均相对误差
    :param true_value: 真实值
    :param pred_value: 预测值
    :return: 平均相对误差
    """
    if np.any(true_value == 0):
        return 0
    return mean_absolute_percentage_error(true_value, pred_value) * 100


def amape(label, pred):
    """
    得到各轴上的平均相对误差
    :param label:
    :param pred:
    :return:
    """
    pred, label = np.array(pred), np.array(label)
    x_pred, y_pred, z_pred = pred[:, 0], pred[:, 1], pred[:, 2]
    x_label, y_label, z_label = label[:, 0], label[:, 1], label[:, 2]
    mean_relative_error_x = mape(x_label, x_pred)
    mean_relative_error_y = mape(y_label, y_pred)
    mean_relative_error_z = mape(z_label, z_pred)

    return mean_relative_error_x, mean_relative_error_y, mean_relative_error_z


def mse(true_value, pred_value):
    """
    计算均方误差
    :param y_true:
    :param y_pred:
    :return:
    """
    error = mean_squared_error(true_value, pred_value)
    return error


def amse(label, pred):
    """
    得到各轴上的均方误差
    :param label:
    :param pred:
    :return:
    """
    pred, label = np.array(pred), np.array(label)
    x_pred, y_pred, z_pred = pred[:, 0], pred[:, 1], pred[:, 2]
    x_label, y_label, z_label = label[:, 0], label[:, 1], label[:, 2]
    mean_squared_error_x = mse(x_label, x_pred)
    mean_squared_error_y = mse(y_label, y_pred)
    mean_squared_error_z = mse(z_label, z_pred)

    return mean_squared_error_x, mean_squared_error_y, mean_squared_error_z


def plot_figure(config, label, pred):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 24))

    plot_single_axis(ax1, pred[:, 0], label[:, 0], 'X')
    plot_single_axis(ax2, pred[:, 1], label[:, 1], 'Y')
    plot_single_axis(ax3, pred[:, 2], label[:, 2], 'Z')

    plt.tight_layout()

    # 保存图像到指定路径
    plt.savefig(os.path.join(config.file_path, "result_plot.png"), dpi=300, bbox_inches='tight')  # dpi 控制分辨率，bbox_inches 确保图像内容完整

    plt.show()


def plot_single_axis(ax, pred, label, axis_label):
    # 主轴：绘制预测值与真实值
    ax.plot(pred, color='red', label=f'Predicted load value in {axis_label}')
    ax.plot(label, color='black', label=f'True load value in {axis_label}')
    ax.autoscale(enable=True, axis='x')
    ax.autoscale(enable=True, axis='y')
    ax.set_xlabel("Data Sequence", fontsize=16)
    ax.set_ylabel("Load(N)", fontsize=16, color='black')
    ax.tick_params(axis='y', labelcolor='black')

    # 计算相对误差
    epsilon = 1e-8  # 极小值，避免分母为0
    relative_error = np.abs((label - pred) / (label + epsilon)) * 100

    # 副轴：绘制相对误差
    ax2 = ax.twinx()
    ax2.plot(relative_error, 'b^', label=f'Relative Error in {axis_label}', markersize=7, markerfacecoloralt='white', fillstyle='right')
    ax2.set_ylabel("Relative Error (%)", fontsize=16, color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    # 设置副轴的上限最小值为5
    current_ylim = ax2.get_ylim()
    if current_ylim[1] < 5:
        ax2.set_ylim(current_ylim[0], 5)
    else:
        ax2.set_ylim(0, current_ylim[1]*1.3)

    # 合并图例并放置在右上角
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')


def inverse_preds(config, preds):
    df = pd.read_csv(config.data_path, header=0).values
    labels = df[:, -3:]
    # 还原后的真实值
    scaled_labels = labels[config.test_index: config.test_index + 370]

    scaler = MinMaxScaler()
    labels = scaler.fit_transform(labels)
    # 将预测值与真实值进行替换
    labels[config.test_index: config.test_index + 370, :] = preds
    # 还原替换后的数据
    labels_scaled = scaler.inverse_transform(labels)
    scaled_preds = labels_scaled[config.test_index: config.test_index + 370, :]
    return scaled_labels, scaled_preds


def save_data_to_excel(config, model, preds, labels, dimension):
    data = {'preds': preds[:, dimension], 'labels': labels[:, dimension]}
    model_name = model.__class__.__name__
    file_path = os.path.join(config.file_path, f"{model_name}_{dimension}.xlsx")
    df = pd.DataFrame(data)
    df.to_excel(file_path, index=False)
    print(f"Saved file to {file_path}")


def save_file(config, model, preds, labels):
    dimensions = [0, 1, 2]

    for dimension in dimensions:
        save_data_to_excel(config, model, preds, labels, dimension)


def l1_regularization(model, l1_lambda):
    l1_loss = torch.tensor(0., requires_grad=True)
    for param in model.parameters():
        l1_loss = l1_loss + torch.sum(torch.abs(param))
    return l1_lambda * l1_loss
