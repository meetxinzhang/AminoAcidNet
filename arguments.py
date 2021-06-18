import configargparse


def build_parser():
    parser = configargparse.ArgParser(default_config_files=['settings.conf'])

    # parser.add_argument('-pdb_dir', default='/media/zhangxin/Raid0/dataset/PP/',
    #                     help='Directory where all protein pdb files exist')
    # parser.add_argument('-pkl_dir', default='media/zhangxin/Raid0/dataset/PP/pkl/')
    # parser.add_argument('-json_dir', default='/media/zhangxin/Raid0/dataset/PP/json/')
    # parser.add_argument('-h5_dir', default='/media/zhangxin/Raid0/dataset/PP/h5/')
    # parser.add_argument('-h5_name', default='h5_dataset_id')
    # parser.add_argument('-cpp_executable', default='preprocess/get_features',
    #                     help='Directory where cpp cpp_executable is located')

    parser.add_argument('-parallel_jobs', default=20, help='Number of threads to use for parallel jobs')
    parser.add_argument('-get_json_files', default=False, help='Whether to fetch json files or not',
                        action='store_true')
    parser.add_argument('-override_h5_dataset', default=True, help='Whether to fetch h5 dataset or not')
    parser.add_argument('-max_neighbors', default=15)
    parser.add_argument('-bind_radius', default=30)
    parser.add_argument('-inputs_padding', default=True)

    # Training setup
    parser.add_argument('--random_seed', default=123, help='Seed for random number generation', type=int)
    parser.add_argument('--epochs', default=100, help='Number of epochs', type=int)
    parser.add_argument('--batch_size', default=3, help='Batch size for training', type=int)
    parser.add_argument('--train', default=0.5, help='Fraction of training data', type=float)
    parser.add_argument('--val', default=0.25, help='Fraction of validation data', type=float)
    parser.add_argument('--test', default=0.25, help='Fraction of test data', type=float)
    parser.add_argument('--testing', help='If only testing the model', action='store_true')

    # Optimizer setup
    parser.add_argument('--lr', default=0.001, help='Learning rate', type=float)

    # Model setup
    parser.add_argument('--h_a', default=64, help='Atom hidden embedding dimension', type=int)
    parser.add_argument('--h_g', default=32, help='Graph hidden embedding dimension', type=int)
    parser.add_argument('--n_conv', default=4, help='Number of convolution layers', type=int)

    # Other features
    parser.add_argument('--save_checkpoints', default=True, help='Stores checkpoints if true', action='store_true')
    parser.add_argument('--print_freq', default=10, help='Frequency of printing updates between epochs', type=int)
    parser.add_argument('--workers', default=20, help='Number of workers for data loading', type=int)

    return parser
