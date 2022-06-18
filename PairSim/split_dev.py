import argparse
import os
import random

import pandas as pd


def main(args):
    random.seed(args.seed)
    full_train_data = pd.read_csv(os.path.join(args.data_dir, 'full_train.tsv'), sep='\t')
    full_samples_num = len(full_train_data)

    dev_indices = random.sample(range(full_samples_num), int(full_samples_num * args.dev_prop))
    train_indices = list(set(range(full_samples_num)) - set(dev_indices))

    train_data = full_train_data.loc[train_indices]
    dev_data = full_train_data.loc[dev_indices]

    print(f'#Train samples: {len(train_data)}\n'
          f'#Dev samples: {len(dev_data)}')

    train_data.to_csv(os.path.join(args.data_dir, 'train.tsv'), sep='\t')
    dev_data.to_csv(os.path.join(args.data_dir, 'dev.tsv'), sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--data_dir', type=str, default='../data/')
    # options
    parser.add_argument('--dev_prop', type=float, default=0.02)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    main(args)
