import numpy as np
from torch.utils.data import TensorDataset
from torch_geometric.data import DataLoader
from tqdm import tqdm
import time
import torch.nn
from src.args import get_args_
from src.dataloader import MyDataloader, MyDataset
from src.trainer import Trainer
from tensorboardX import SummaryWriter
from src.kan import *
from src.mamba import *
from src.mymodel import *
from src.models import *
from src.utils import *

config = get_args_()
# 消融实验模型字典
ablation_models = {
    # "Baseline": MultiScaleAttentionTransformer(embed_size=64, num_heads=8, num_layers=1, forward_expansion=4,
    #                                            dropout=0.2, output_dim=3),
    "SingleScale": AblationSingleScale(embed_size=64, num_heads=8, num_layers=1, forward_expansion=4,
                                       dropout=0.2, output_dim=3),
    "NoAttention": AblationNoAttention(embed_size=64, num_heads=8, num_layers=1, forward_expansion=4,
                                       dropout=0.2, output_dim=3),
    "NoPositionEncoding": AblationNoPositionEncoding(embed_size=64, num_heads=8, num_layers=1,
                                                     forward_expansion=4, dropout=0.2, output_dim=3),
    # "SingleDirectionOutputX": AblationSingleDirectionOutputX(embed_size=64, num_heads=8, num_layers=1,
    #                                                         forward_expansion=4, dropout=0.2, output_dim=1),
    # "SingleDirectionOutputY": AblationSingleDirectionOutputY(embed_size=64, num_heads=8, num_layers=1,
    #                                                          forward_expansion=4, dropout=0.2, output_dim=1),
    # "SingleDirectionOutputZ": AblationSingleDirectionOutputZ(embed_size=64, num_heads=8, num_layers=1,
    #                                                          forward_expansion=4, dropout=0.2, output_dim=1)
}


def main():
    print("Loading data...")
    dataset = MyDataset(config.data_path)
    data_loader = MyDataloader(config=config, dataset=dataset)
    train_loader, test_loader = data_loader.train_loader, data_loader.test_loader

    print("Loading model...")
    start_time = time.time()

    model = MultiScaleAttentionTransformer(embed_size=64, num_heads=8, num_layers=1, forward_expansion=4,
                                           dropout=0.2, output_dim=3)


    writer = SummaryWriter(log_dir=config.log_dir)

    trainer = Trainer(config=config)

    trainer.train(model, train_loader, test_loader, writer)
    trainer.test(model, test_loader)

    writer.close()
    end_time = time.time()

    print("训练时间:", end_time - start_time)


def ablation_experiment(config, ablation_models):
    print("Starting ablation experiments...")
    dataset = MyDataset(config.data_path)
    data_loader = MyDataloader(config=config, dataset=dataset)
    train_loader, test_loader = data_loader.train_loader, data_loader.test_loader

    results = {}

    for model_name, model in ablation_models.items():
        print(f"Running experiment for model: {model_name}")
        writer = SummaryWriter(log_dir=f"{config.log_dir}/{model_name}")
        trainer = Trainer(config=config)
        trainer.train(model, train_loader, test_loader, writer)
        performance = trainer.test(model, test_loader)
        results[model_name] = performance
        writer.close()

    # 保存实验结果
    with open(f"{config.log_dir}/ablation_results.txt", "w") as f:
        for model_name, metrics in results.items():
            f.write(f"{model_name}:\n")
            for metric, value in metrics.items():
                f.write(f"  {metric}: {value}\n")
            f.write("\n")
    print(results)
    print("Ablation experiments completed.")


def leave_one_out(config):
    print("==========开始逐一验证（Leave-One-Out Cross-Validation）==========")
    dataset = MyDataset(config.data_path)
    all_data = np.array(dataset.data, dtype=np.float32)
    all_label = np.array(dataset.label, dtype=np.float32)

    scaler = MinMaxScaler()
    all_data = scaler.fit_transform(all_data)  # 标准化数据

    for test_index in range(0, len(all_data), 370):
        model = MultiScaleAttentionTransformer(embed_size=64, num_heads=8, num_layers=1, forward_expansion=4,
                                               dropout=0.2, output_dim=3).to(config.device)
        # model = TransformerModel(input_size=config.input_size, output_size=config.output_size, num_hidden_layers=2,
        #                          d_model=64, nhead=8).to(config.device)

        train_data = np.concatenate((all_data[:test_index], all_data[test_index + 370:]), axis=0)
        test_data = all_data[test_index:test_index + 370]

        train_label = np.concatenate((all_label[:test_index], all_label[test_index + 370:]), axis=0)
        test_label = all_label[test_index:test_index + 370]

        train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32).to(config.device),
                                      torch.tensor(train_label, dtype=torch.float32).to(config.device))
        test_dataset = TensorDataset(torch.tensor(test_data, dtype=torch.float32).to(config.device),
                                     torch.tensor(test_label, dtype=torch.float32).to(config.device))

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

        writer = SummaryWriter(log_dir=config.log_dir)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
        criterion = nn.MSELoss()

        best_model_path = None
        min_error = float('inf')

        print('==========开始训练==========')
        for epoch in range(config.epochs):
            model.train()
            train_loss = 0.0
            tbar = tqdm(train_loader, total=len(train_loader), desc=f'训练 Epoch {epoch + 1}/{config.epochs}')
            for idx, (feature, label) in enumerate(tbar):
                optimizer.zero_grad()
                pred = model(feature)
                loss = criterion(pred, label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                writer.add_scalar(f'训练/损失_fold_{test_index // 370 + 1}', loss.item(),
                                  epoch * len(train_loader) + idx)
                tbar.set_postfix(loss=f'{train_loss / (idx + 1):.8f}')

            model.eval()
            valid_loss = 0.0
            val_preds, val_labels = [], []
            vbar = tqdm(test_loader, total=len(test_loader), desc='验证中')
            with torch.no_grad():
                for idx, (feature, label) in enumerate(vbar):
                    pred = model(feature)
                    loss = criterion(pred, label)
                    valid_loss += loss.item()
                    val_preds.append(pred.cpu().numpy())
                    val_labels.append(label.cpu().numpy())
                    writer.add_scalar(f'验证/损失_fold_{test_index // 370 + 1}', loss.item(),
                                      epoch * len(test_loader) + idx)
                    vbar.set_postfix(loss=f'{valid_loss / (idx + 1):.8f}')
            scheduler.step(valid_loss / len(test_loader))

            val_preds = np.concatenate(val_preds, axis=0)
            val_labels = np.concatenate(val_labels, axis=0)
            x_errors, y_errors, z_errors = amape(val_preds, val_labels)

            if x_errors + y_errors + z_errors < min_error:
                min_error = x_errors + y_errors + z_errors
                best_model_dir = f'{config.model_dir}/{model.__class__.__name__}'
                best_model_path = f'{best_model_dir}/{model.__class__.__name__}_work_{test_index // 370 + 1}.pth'

                if not os.path.exists(best_model_dir):
                    os.makedirs(best_model_dir)

                torch.save(model.state_dict(), best_model_path)

        if best_model_path is not None:
            model.load_state_dict(torch.load(best_model_path))
        model.eval()
        preds, labels = [], []
        tbar = tqdm(test_loader, total=len(test_loader), desc='测试中')
        with torch.no_grad():
            for idx, (feature, label) in enumerate(tbar):
                pred = model(feature)
                preds.append(pred.cpu().numpy())
                labels.append(label.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        labels = np.concatenate(labels, axis=0)

        # 使用原始的缩放器进行逆变换
        df = pd.read_csv(config.data_path, header=0).values
        origin_labels = df[:, -3:]
        scaled_labels = origin_labels[test_index: test_index + 370]
        origin_labels = scaler.fit_transform(origin_labels)
        origin_labels[test_index: test_index + 370, :] = preds
        labels_scaled = scaler.inverse_transform(origin_labels)
        scaled_preds = labels_scaled[test_index:test_index + 370, :]

        x_errors, y_errors, z_errors = amape(scaled_labels, scaled_preds)
        x_mae_errors, y_mae_errors, z_mae_errors = amae(scaled_preds, scaled_labels)
        x_mse_errors, y_mse_errors, z_mse_errors = amse(scaled_preds, scaled_labels)
        plot_figure(config, scaled_labels, scaled_preds)
        if not os.path.exists('result_3'):
            os.makedirs('result_3')
        with open(f'result_3/test_results_work_{test_index // 370 + 1}.txt', 'w') as f:
            f.write("测试结果:\n预测值\t实际值\n")
            for pred, label in zip(scaled_preds, scaled_labels):
                f.write(f"{pred}\t{label}\n")
            f.write("\n性能指标:\n")
            f.write(f"x轴平均相对误差={x_errors:.4f}%\n")
            f.write(f"y轴平均相对误差={y_errors:.4f}%\n")
            f.write(f"z轴平均相对误差={z_errors:.4f}%\n")
            f.write(f"x轴平均绝对误差={x_mae_errors:.8f}\n")
            f.write(f"y轴平均绝对误差={y_mae_errors:.8f}\n")
            f.write(f"z轴平均绝对误差={z_mae_errors:.8f}\n")
            f.write(f"x轴均方误差={x_mse_errors:.8f}\n")
            f.write(f"y轴均方误差={y_mse_errors:.8f}\n")
            f.write(f"z轴均方误差={z_mse_errors:.8f}\n")

    print("==========逐一验证完成（Leave-One-Out Cross-Validation Done）==========")


if __name__ == '__main__':
    main()

    # leave_one_out(config) # 留一法

    # ablation_experiment(config, ablation_models)  # 消融实验


