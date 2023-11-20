from pred import load_model, read_input_file, convert_input_file_to_tensor_dataset
import torch
from accelerate import Accelerator
accelerator = Accelerator()
from data_loader import get_labels
device = accelerator.device
from tokenizer import KoBertTokenizer
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from tqdm import tqdm
import numpy as np

max_seq_len = 50
batch_size = 1
pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
model_addr = 'monologg/kobert'
model_type = "kobert"
output_file = "./result/pred/kobert_pred.txt"
tokenizer = KoBertTokenizer.from_pretrained(model_addr)
model = load_model(device)
lines = read_input_file()
label_lst = get_labels()
dataset = convert_input_file_to_tensor_dataset(lines, max_seq_len, tokenizer, pad_token_label_id)
sampler = SequentialSampler(dataset)
data_loader_ = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
all_slot_label_mask = None
preds = None

model, data_loader_ = accelerator.prepare(model, data_loader_)

for batch in tqdm(data_loader_, desc="Predicting"):
        batch = tuple(t for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": None}
            if model_type != "distilkobert":
                inputs["token_type_ids"] = batch[2]
            outputs = model(**inputs)
            logits = outputs[0]

            if preds is None:
                preds = logits.detach().cpu().numpy()
                all_slot_label_mask = batch[3].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                all_slot_label_mask = np.append(all_slot_label_mask, batch[3].detach().cpu().numpy(), axis=0)

preds = np.argmax(preds, axis=2)
slot_label_map = {i: label for i, label in enumerate(label_lst)}
preds_list = [[] for _ in range(preds.shape[0])]

for i in range(preds.shape[0]):
    for j in range(preds.shape[1]):
        if all_slot_label_mask[i, j] != pad_token_label_id:
            preds_list[i].append(slot_label_map[preds[i][j]])
# Write to output file
with open(output_file, "w", encoding="utf-8") as f:
    for words, preds in zip(lines, preds_list):
        line = ""
        for word, pred in zip(words, preds):
            if pred == 'O':
                line = line + word + " "
            else:
                line = line + "[{}:{}] ".format(word, pred)
        f.write("{}\n".format(line.strip()))
print("Prediction Done!")