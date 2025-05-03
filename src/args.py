import argparse

#tensorboard --logdir=./works/work_x
def get_args_():
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('--mode', default='train', type=str, help='train or test')
    parser.add_argument('--test_index', default=6660, type=int, help='test index number')
    parser.add_argument('--data_path', default=R'Dataset/26.6mm_1.csv',
                        type=str, help='path of the dataset')
    parser.add_argument('--file_path', default=R'files_new',
                        type=str, help='path of the files')
    parser.add_argument('--model_dir', default=R'model_weights',
                        type=str, help='dir of the model weights')
    parser.add_argument('--log_dir', default=R'logs', type=str,
                        help='dir of the log')
    parser.add_argument('--min_error', default=float('inf'), type=float, help='min error in validation set')
    parser.add_argument('--device', default='cuda', type=str, help='device for training')
    parser.add_argument('--input_size', default=8, type=int, help='size of the input')
    parser.add_argument('--output_size', default=1, type=int, help='size of the output')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--epochs', default=300, type=int, help='number of epochs')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')

    args = parser.parse_args()

    return args
