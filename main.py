import argparse
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW

from PairSim.dataloader import MyDataset
from PairSim.model import PairSimNet, PairSimNet2
from PairSim.trainer import Trainer


def main(args):
    torch.manual_seed(args.seed)

    # load data
    if args.do_train:
        train_dataset = MyDataset(args, data_path=os.path.join(args.data_dir, 'train.tsv'))
        print(f'Loaded #Train samples: {len(train_dataset)}')
        dev_dataset = MyDataset(args, data_path=os.path.join(args.data_dir, 'dev.tsv'))
        print(f'Loaded #Dev samples: {len(dev_dataset)}')
        train_dataloader, dev_dataloader = [DataLoader(_, batch_size=args.batch_size, shuffle=True)
                                             for _ in [train_dataset, dev_dataset]]
    if args.do_eval:
        dev_dataset = MyDataset(args, data_path=os.path.join(args.data_dir, 'dev.tsv'))
        print(f'Loaded #Dev samples: {len(dev_dataset)}')
        dev_dataloader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, shuffle=False)
    if args.do_predict:
        test_dataset = MyDataset(args, data_path=os.path.join(args.data_dir, 'test.tsv'))
        print(f'Loaded #Test samples: {len(test_dataset)}')
        test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False)
    # initialize tools
    if args.cls_method == 'one-pass':
        model = PairSimNet(args, model_name=args.model_name)
    elif args.cls_method == 'two-pass':
        model = PairSimNet2(args, model_name=args.model_name)
    else:
        raise ValueError(f'cls_method {args.cls_method} is not supported.')

    if args.load_weights_from is not None:
        model.load_state_dict(torch.load(args.load_weights_from))
    if args.use_cuda:
        device = torch.device('cuda')
        model = nn.DataParallel(model)
    else:
        device = torch.device('cpu')
    model.to(device)

    trainer = Trainer(args, device=device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCELoss()

    # train
    if args.do_train:
        trainer.train(
            train_dataloader=train_dataloader,
            dev_dataloader=dev_dataloader,
            model=model,
            model_name=args.model_name,
            cls_method=args.cls_method,
            optimizer=optimizer,
            criterion=criterion
        )

    # evaluate
    if args.do_eval:
        eval_loss, eval_acc, eval_results = trainer.evaluate(
            dataloader=dev_dataloader,
            model=model,
            model_name=args.model_name,
            cls_method=args.cls_method,
            criterion=criterion
        )
        if args.load_weights_from is not None:
            ckpt_name = args.load_weights_from.split('/')[-1][:-3]
            with open(os.path.join(args.result_dir, f"eval_result_{ckpt_name}.json"), 'w') as f:
                json.dump(eval_results, f)

        # predict
    if args.do_predict:
        pred_results = trainer.predict(
            dataloader=test_dataloader,
            model=model,
            model_name=args.model_name,
            cls_method=args.cls_method,
        )
        if args.load_weights_from is not None:
            ckpt_name = args.load_weights_from.split('/')[-1][:-3]
            with open(os.path.join(args.result_dir, f"pred_result_{ckpt_name}.json"), 'w') as f:
                json.dump(pred_results, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--exp_name', type=str, default='bert')
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
    parser.add_argument('--result_dir', type=str, default='./PairSim/results/')
    parser.add_argument('--load_weights_from', type=str, default=None)
    # options
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--do_predict', action='store_true')
    # training setting
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--report_steps', type=int, default=100)
    parser.add_argument('--save_steps', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    # model setting
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--cls_method', type=str, default='one-pass',
                        choices=['one-pass', 'two-pass'])
    parser.add_argument('--max_input_length', type=int, default=128)

    args = parser.parse_args()

    args.result_dir = os.path.join(args.result_dir, f'{args.exp_name}/')
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, f'{args.exp_name}/')
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(args.checkpoint_dir, 'logging/'), exist_ok=True)

    main(args)
