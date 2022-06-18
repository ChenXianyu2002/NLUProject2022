import json
import os

import numpy as np
import torch
from tqdm import tqdm


class Trainer(object):
    def __init__(self, args, device='cpu'):
        self.args = args
        self.max_epochs = args.max_epochs
        self.report_steps = args.report_steps
        self.save_steps = args.save_steps
        self.checkpoint_dir = args.checkpoint_dir

        self.device = device

    def step(self, batch_data, cls_method, model_name, model, ):
        if cls_method == 'one-pass':
            eid, sent_item, labels = batch_data
            labels = labels.to(self.device)
            if 'roberta' in model_name:
                input_ids, attn_mask = \
                    sent_item['input_ids'].squeeze(1).to(self.device), \
                    sent_item['attention_mask'].squeeze(1).to(self.device)
                logits = model(input_ids, attn_mask)  # [b, 1]
            else:
                input_ids, attn_mask, token_type_ids = \
                    sent_item['input_ids'].squeeze(1).to(self.device), \
                    sent_item['attention_mask'].squeeze(1).to(self.device), \
                    sent_item['token_type_ids'].squeeze(1).to(self.device)
                logits = model(input_ids, attn_mask, token_type_ids)  # [b, 1]
        elif cls_method == 'two-pass':
            eid, sent1_item, sent2_item, labels = batch_data
            labels = labels.to(self.device)
            if 'roberta' in model_name:
                input_ids1, attn_mask1 = \
                    sent1_item['input_ids'].squeeze(1).to(self.device), \
                    sent1_item['attention_mask'].squeeze(1).to(self.device)
                input_ids2, attn_mask2 = \
                    sent2_item['input_ids'].squeeze(1).to(self.device), \
                    sent2_item['attention_mask'].squeeze(1).to(self.device)
                logits = model(input_ids1, input_ids2, attn_mask1, attn_mask2)
            else:
                input_ids1, attn_mask1, token_type_ids1 = \
                    sent1_item['input_ids'].squeeze(1).to(self.device), \
                    sent1_item['attention_mask'].squeeze(1).to(self.device), \
                    sent1_item['token_type_ids'].squeeze(1).to(self.device)
                input_ids2, attn_mask2, token_type_ids2 = \
                    sent2_item['input_ids'].squeeze(1).to(self.device), \
                    sent2_item['attention_mask'].squeeze(1).to(self.device), \
                    sent2_item['token_type_ids'].squeeze(1).to(self.device)
                logits = model(input_ids1, input_ids2, attn_mask1, attn_mask2, token_type_ids1, token_type_ids2)
        else:
            raise ValueError(f'cls_method {cls_method} is not supported.')
        return logits, labels, eid

    def train(self, train_dataloader, dev_dataloader, model, model_name, cls_method, optimizer, criterion):
        model.train()
        total_steps = 0
        best_acc, running_loss = 0.0, 0.0
        for epoch in range(self.max_epochs):
            for _, batch_data in enumerate(tqdm(train_dataloader)):
                total_steps += 1
                logits, labels, _ = self.step(batch_data, cls_method, model_name, model)

                optimizer.zero_grad()
                loss = criterion(logits.squeeze(-1), labels.float())
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if total_steps % self.report_steps == 0:
                    print(f'Epoch#{epoch}, total steps#{total_steps}, loss:{running_loss / self.report_steps}')
                    running_loss = 0.0
                if total_steps % self.save_steps == 0:
                    print('-' * 20, "Evaluating", '-' * 20)
                    eval_loss, eval_acc, results = self.evaluate(dev_dataloader, model, model_name, cls_method,
                                                                 criterion)
                    eval_loss, eval_acc = round(eval_loss, 4), round(eval_acc, 4)
                    print(f'Evaluate result: '
                          f'Epoch#{epoch}, total steps#{total_steps}, loss:{eval_loss}, acc: {eval_acc}')
                    if eval_acc >= best_acc:
                        print(f"Get new Best Result:{eval_acc},Saving...")
                        best_acc = eval_acc
                        save_path = os.path.join(self.checkpoint_dir,
                                                 f'ckpt_ep{epoch}_step{total_steps}_loss{eval_loss}_acc{eval_acc}.pt')
                        torch.save(model.module.state_dict(), save_path)
                        print("Saved!")
                    with open(os.path.join(self.checkpoint_dir, 'logging/',
                                           f'log_ep{epoch}_step{total_steps}_loss{loss}_acc{eval_acc}.json'), 'w') as f:
                        eval_dict = {
                            'eval_loss': eval_loss,
                            'eval_acc': eval_acc,
                            'results': results
                        }
                        json.dump(eval_dict, f)
        torch.cuda.empty_cache()

    def evaluate(self, dataloader, model, model_name, cls_method, criterion):
        running_loss = 0.

        eid_list, label_list, pred_list = [], [], []
        results = []
        model.eval()
        for _, batch_data in enumerate(tqdm(dataloader)):
            logits, labels, eid = self.step(batch_data, cls_method, model_name, model)

            loss = criterion(logits.squeeze(-1), labels.float())
            running_loss += loss.item()
            probs = logits.unsqueeze(-1).detach().cpu().numpy()
            preds = probs.copy()
            preds[preds > 0.5] = 1
            preds[preds <= 0.5] = 0
            preds = preds.squeeze()
            eid = eid.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            preds = preds.detach().cpu().numpy()
            eid_list.extend(eid)
            label_list.extend(labels)
            pred_list.extend(preds)
        running_loss /= len(dataloader)
        accuracy = float(np.mean(np.equal(pred_list, label_list)))
        for eid, label, pred in zip(eid_list, label_list, pred_list):
            results.append({
                'eid': int(eid),
                'label': int(label),
                'pred': int(pred)
            })
        model.train()
        return running_loss, accuracy, results

    def predict(self, dataloader, model, model_name, cls_method
                ):
        eid_list, pred_list = [], []
        results = []
        model.eval()
        for _, batch_data in enumerate(tqdm(dataloader)):
            logits, _, eid = self.step(batch_data, cls_method, model_name, model)
            probs = logits.unsqueeze(-1).detach().cpu().numpy()
            preds = probs.copy()
            preds[preds > 0.5] = 1
            preds[preds <= 0.5] = 0
            preds = preds.squeeze()
            eid = eid.detach().cpu().numpy()
            preds = preds.detach().cpu().numpy()
            eid_list.extend(eid)
            pred_list.extend(preds)

        for eid, pred in zip(eid_list, pred_list):
            results.append({
                'eid': int(eid),
                'pred': int(pred)
            })

        return results
