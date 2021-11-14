import argparse
from utility import run
import logging


def parse_args():
    parser = argparse.ArgumentParser()

    # General params
    parser.add_argument('--path', type=str, default='data/retail-rocket/events.csv',
                        help='Path to data file.')
    parser.add_argument('--data_type', type=str, default='retail-rocket',
                        help='The type of the dataset used. Currently supported values are [retail-rocket, recsys15].')
    parser.add_argument('--model_type', type=str, default='caser',
                        help='The type of model to be used. '
                             'Currently supported values are [caser, nextitnet, gru4rec, sas4rec, random].')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='Use False for CPU and True for CUDA computation.')
    parser.add_argument('-top_k', nargs='+', default=[5, 10, 15, 20], help='Top k items for ndcg and hr computation.')
    parser.add_argument('--validation_interval', type=int, default=None, help='Governs validation frequency.')

    # Dev-mode params
    parser.add_argument('--dev_mode', type=bool, default=False,
                        help='True to run the application in dev mode with a limited sample size. '
                             'False for full-scale runs.')
    parser.add_argument('--dev_mode_path', type=str, default='data/retail-rocket/samples/sample_0.01.csv',
                        help='Path to sample file to be used in dev mode. Parameter ignored if dev_mode set to False.')

    # Generic model params
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of training epochs.')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Amount of dropout to be used in fully-connected output layers of classification models.')
    parser.add_argument('--max_seq_len', type=float, default=10, help='The maximum length of an interaction sequence.')
    parser.add_argument('--batch_size', type=int, default=256, help='Training batch size.')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Dimensionality of the hidden layers.')
    parser.add_argument('--lr', type=float, default=0.01, help='Training learning rate.')

    # CASER params
    parser.add_argument('-caser_horizontal_filters', nargs='+', default=[2, 3, 4, 5],
                        help='List of filter sizes for Caser model.')

    # NextItItem params
    parser.add_argument('--nextitnet_kernel_size', type=int, default=3, help='The kernel size parameter for NextItNet')
    parser.add_argument('--nextitnet_dilated_channels', type=int, default=512,
                        help='The dilated channel size for NextItNet')
    parser.add_argument('--nextitnet_dilations', nargs='+', default=[1, 2, 4, 8, 1, 2, 4, 8],
                        help='The dilations for the individual convolutions of NextItNet')

    # GRU4Rec params
    # no specific parameters currently required

    # SAS4Rec params
    parser.add_argument('--sas4rec_num_sablocks', type=int, default=1,
                        help='The number of self-attention blocks in the SAS4Rec model.')
    parser.add_argument('--sas4rec_positionwise_feedforward_dim', type=int, default=64,
                        help='The positionwise feed-forward network inner dimension for the SAS4Rec model.')

    return parser.parse_args()


if __name__ == '__main__':
    logger = logging.getLogger('sequential-recommendation')
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(level=logging.INFO)

    args = parse_args()
    logger.log(msg=f'Parameters: {args}', level=logging.INFO)
    run(args)
