import torch.onnx
import torch.nn as nn
from tqdm import tqdm
from src.utils import *
import onnxruntime
import time


class Trainer:
    def __init__(self, config):
        self.config = config

    def train(self, model, train_dataloader, test_dataloader, writer):
        # prepare optimizer and criterion
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
        criterion = nn.MSELoss()
        model.train()
        model = model.to(self.config.device)

        print('==========Start Training==========')
        for epoch in range(self.config.epochs):
            # train epoch
            self.train_epoch(epoch, model, train_dataloader, optimizer, criterion, writer)

            # val epoch
            self.val_epoch(epoch, model, test_dataloader, criterion, writer, scheduler)
        print('==========Training Done==========')

    def train_epoch(self, epoch, model, train_dataloader, optimizer, criterion, writer):
        model.train()
        train_loss = 0.0
        tbar = tqdm(train_dataloader, total=len(train_dataloader), desc='Training')
        for idx, (feature, label) in enumerate(train_dataloader):
            feature, label = feature.to(self.config.device), label.to(self.config.device)
            optimizer.zero_grad()
            pred = model(feature)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            writer.add_scalar('Train/Loss', loss.item(), epoch)

            tbar.set_description(f'Epoch [{epoch + 1}/{self.config.epochs}]')
            tbar.set_postfix(epoch=epoch + 1, loss=f'{train_loss / (idx + 1):.8f}')
            tbar.update()

    @torch.no_grad()
    def val_epoch(self, epoch, model, test_dataloader, criterion, writer, scheduler):
        model.eval()
        preds, labels = [], []
        valid_loss = 0.0
        vbar = tqdm(test_dataloader, total=len(test_dataloader), desc='Validating')
        for idx, (feature, label) in enumerate(test_dataloader):
            feature, label = feature.to(self.config.device), label.to(self.config.device)
            pred = model(feature)

            preds.append(pred.cpu().numpy())
            labels.append(label.cpu().numpy())

            loss = criterion(pred, label)
            valid_loss += loss.item()

            writer.add_scalar('Val/Loss', loss.item(), epoch)

            vbar.set_description(f'Epoch [{epoch + 1}/{self.config.epochs}]')
            vbar.set_postfix(epoch=epoch + 1, loss=f'{valid_loss / (idx + 1):.8f}')
            vbar.update()

        scheduler.step(valid_loss / len(test_dataloader))

        preds = [x for list in preds for x in list]
        labels = [y for list in labels for y in list]
        x_errors, y_errors, z_errors = amape(preds, labels)

        # save model
        if x_errors + y_errors + z_errors < self.config.min_error:
            #self.config.min_error = x_errors + y_errors + z_errors
            model_name = model.__class__.__name__
            torch.save(model.state_dict(), self.config.model_dir + f'{model_name}.pth')

    @torch.no_grad()
    def test(self, model, test_dataloader):
        print("==========Start Testing==========")
        model = model.to(self.config.device)
        model_name = model.__class__.__name__
        model.load_state_dict(torch.load(self.config.model_dir + f'{model_name}.pth'))
        model.eval()
        preds, labels = [], []

        tbar = tqdm(test_dataloader, total=len(test_dataloader), desc='Testing')
        for idx, dl in enumerate(tbar):
            feature, label = dl
            feature = feature.to(self.config.device)
            label = label.to(self.config.device)
            pred = model(feature)
            preds.append(pred.cpu().numpy())
            labels.append(label.cpu().numpy())
            tbar.update()

        preds = [x for list in preds for x in list]
        labels = [y for list in labels for y in list]

        scaled_labels, scaled_preds = inverse_preds(self.config, preds)

        x_errors, y_errors, z_errors = amape(scaled_labels, scaled_preds)
        print(
            f'x轴平均相对误差={x_errors:.4f}%',
            f'y轴平均相对误差={y_errors:.4f}%',
            f'z轴平均相对误差={z_errors:.4f}%')

        x_mae_errors, y_mae_errors, z_mae_errors = amae(scaled_preds, scaled_labels)
        print(
            f'x轴平均绝对误差={x_mae_errors:.8f}',
            f'y轴平均绝对误差={y_mae_errors:.8f}',
            f'z轴平均绝对误差={z_mae_errors:.8f}')

        x_mse_errors, y_mse_errors, z_mse_errors = amse(scaled_preds, scaled_labels)
        print(
            f'x轴均方根误差={x_mse_errors:.8f}',
            f'y轴均方根误差={y_mse_errors:.8f}',
            f'z轴均方根误差={z_mse_errors:.8f}')

        save_file(self.config, model, scaled_preds, scaled_labels)
        plot_figure(self.config, scaled_labels, scaled_preds)

        print('==========Testing Done==========')


def convert_onnx(self, model, test_dataloader):
    model = model.to(self.config.device)
    model_name = model.__class__.__name__
    model_path = f'{self.config.model_dir}{model_name}.pth'

    onnx_file_name = f'{model_name}.onnx'

    model.load_state_dict(torch.load(model_path, map_location=self.config.device))
    model.eval()

    for data, label in test_dataloader:
        torch.onnx.export(model,
                          data,
                          onnx_file_name,
                          verbose=False,
                          keep_initializers_as_inputs=False,
                          opset_version=14,
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}},
                          do_constant_folding=True,

                          )
        print("onnx export complete")
        break


def onnxruntime_test(self, model, test_dataloader):
    print("Start onnx runtime")
    onnx_path = f'{model.__class__.__name__}.onnx'
    session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    start_time = time.time()

    preds, labels = [], []
    tbar = tqdm(test_dataloader, total=len(test_dataloader), desc='Testing')
    for idx, dl in enumerate(tbar):
        feature, label = dl
        feature = feature.to(self.config.device)
        label = label.to(self.config.device)

        feature_np = feature.cpu().numpy()

        pred = session.run(None, {input_name: feature_np})
        preds.append(pred[0])
        labels.append(label.cpu().numpy())
        tbar.update()

    preds = np.concatenate(preds, axis=0)
    scaled_labels, scaled_preds = inverse_preds(self.config, preds)

    x_errors, y_errors, z_errors = amape(scaled_labels, scaled_preds)
    print(
        f'x轴平均相对误差={x_errors:.4f}%',
        f'y轴平均相对误差={y_errors:.4f}%',
        f'z轴平均相对误差={z_errors:.4f}%')

    end_time = time.time()
    # 计算时间间隔
    time_interval = end_time - start_time
    # 将时间间隔转换为总秒数
    total_seconds = time_interval.total_seconds()

    # 将总秒数转换为时分秒格式
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # 输出格式化的时间间隔
    print(f"Average response time cost: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
