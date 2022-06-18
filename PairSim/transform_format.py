import argparse
import json
import os

import pandas as pd


def main(args):
    with open(os.path.join(args.result_dir, args.input_file)) as f:
        results = json.load(f)
    data = [[item['eid'], item['pred']] for item in results]
    df = pd.DataFrame(data, columns=['Id', 'Category'])
    df.to_csv(os.path.join(args.result_dir, args.output_file), sep=',', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--exp_name', type=str, default='diff-simcse-sts-one-pass')
    parser.add_argument('--result_dir', type=str, default='./results/')
    parser.add_argument('--input_file', type=str, default='pred_result_ckpt_ep2_step7800_loss0.2515_acc0.9091.json')
    parser.add_argument('--output_file', type=str, default='submission_ckpt_ep2_step7800_loss0.2515_acc0.9091.csv')
    args = parser.parse_args()

    args.result_dir = os.path.join(args.result_dir, f'{args.exp_name}/')
    os.makedirs(args.result_dir, exist_ok=True)

    main(args)
