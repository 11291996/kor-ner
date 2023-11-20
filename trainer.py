import os
import shutil
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from data_loader import get_labels
    
from utils import MODEL_CLASSES, show_report, compute_metrics

pred_dir = "./result/pred"
model_dir = "./result/model"

class Trainer(object):
    def __init__(self, model_type, model_addr, batch_size, pred = False, train_dataset=None, dev_dataset=None, eval_dataset=None):
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.eval_dataset = eval_dataset
        self.model_type = model_type
        self.model_addr = model_addr
        self.pred = pred

        self.label_lst = get_labels()
        self.num_labels = len(self.label_lst)
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index

        self.config_class, self.model_class, _ = MODEL_CLASSES[self.model_type]

        self.config = self.config_class.from_pretrained(self.model_addr,
                                                        num_labels=self.num_labels,
                                                        finetuning_task="naver-ner",
                                                        id2label={str(i): label for i, label in enumerate(self.label_lst)},
                                                        label2id={label: i for i, label in enumerate(self.label_lst)})
        self.model = self.model_class.from_pretrained(self.model_addr, config=self.config)

        #apply accelerater
        from accelerate import Accelerator
        self.accelerator = Accelerator()
        self.device =  self.accelerator.device
        self.model.to(self.device)

        self.test_texts = None
        if self.pred:
            self.test_texts = get_test_texts()
            # Empty the original prediction files
            if os.path.exists(pred_dir):
                shutil.rmtree(pred_dir)

    def train(self, batch_size, max_steps, epochs, acc_step, weight_decay, lr, adam_epsilon, warmup_steps, max_grad_norm):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=batch_size)

        if max_steps > 0:
            t_total = max_steps
            epochs = max_steps // (len(train_dataloader) // acc_step) + 1
        else:
            t_total = len(train_dataloader) // acc_step * epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr = lr, eps = adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

        # Train!
        print("***** Running training *****")
        print("  Num examples = ", len(self.train_dataset))
        print("  Num Epochs = ", epochs)
        print("  Total train batch size = ", batch_size)
        print("  Gradient Accumulation steps = ", acc_step)
        print("  Total optimization steps = ", t_total)

        global_step = 0
        cur_eval_loss = float("inf")
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(epochs), desc="Epoch")

        train_dataloader, optimizer, scheduler, self.model = \
        self.accelerator.prepare(train_dataloader, scheduler, optimizer, self.model)
            
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]}
                if self.model_type != 'distilkobert':
                    inputs['token_type_ids'] = batch[2]
                outputs = self.model(**inputs)
                loss = outputs[0]

                if acc_step > 1:
                    loss = loss / acc_step

                self.accelerator.backward(loss)

                tr_loss += loss.item()
                if (step + 1) % acc_step == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1
                
            eval_result = self.evaluate("eval", global_step, batch_size = batch_size)["loss"]
            
            if cur_eval_loss > eval_result:
                self.save_model()
            
            cur_eval_loss = eval_result

            if 0 < max_steps < global_step:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step

    def evaluate(self, mode, step, batch_size):
        if mode == 'eval':
            dataset = self.eval_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and eval dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size)

        eval_dataloader = self.accelerator.prepare(eval_dataloader)

        # Eval!
        print("***** Running evaluation on dataset *****", mode)
        print("  Num examples = ", len(dataset))
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'labels': batch[3]}
                if self.model_type != 'distilkobert':
                    inputs['token_type_ids'] = batch[2]
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # Slot prediction
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        # Slot result
        preds = np.argmax(preds, axis=2)
        slot_label_map = {i: label for i, label in enumerate(self.label_lst)}
        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != self.pad_token_label_id:
                    out_label_list[i].append(slot_label_map[out_label_ids[i][j]])
                    preds_list[i].append(slot_label_map[preds[i][j]])

        if self.pred:
            if not os.path.exists(pred_dir):
                os.mkdir(pred_dir)

            with open(os.path.join(pred_dir, "pred_{}.txt".format(step)), "w", encoding="utf-8") as f:
                for text, true_label, pred_label in zip(self.test_texts, out_label_list, preds_list):
                    for t, tl, pl in zip(text, true_label, pred_label):
                        f.write("{} {} {}\n".format(t, tl, pl))
                    f.write("\n")

        result = compute_metrics(out_label_list, preds_list)
        results.update(result)

        print("***** Eval results *****")
        for key in sorted(results.keys()):
            print(key, " ", str(results[key]))
        print("\n" + show_report(out_label_list, preds_list))  # Get the report for each tag result

        return results

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(model_dir)
