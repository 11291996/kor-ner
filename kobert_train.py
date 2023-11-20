#Training kobert 
#from KoBERT_NER.tokenization_kobert import KoBertTokenizer
from tokenizer import KoBertTokenizer
model_addr = 'monologg/kobert'
#get tokenizer
tokenizer = KoBertTokenizer.from_pretrained(model_addr)
#creating datasets
from data_loader import load_and_cache_examples
path = {"train_file":"ner_train.tsv", "dev_file":"", "eval_file":"ner_test.tsv", "label_file":"ner_label.txt"}
max_seq_len = 50
train_data = load_and_cache_examples(path, max_seq_len, tokenizer, "train")
eval_data = load_and_cache_examples(path, max_seq_len, tokenizer, "eval")
dev_data = None
#creating a trainer 
model_type = "kobert"
from trainer import Trainer 
pred = False
kobert_trainer = Trainer(model_type, model_addr, pred, train_dataset=train_data, dev_dataset=dev_data, eval_dataset=eval_data)
#set hyperparameters
batch_size = 256
max_steps = -1 #calculates epoch via steps if not -1 
epochs = 3.0
acc_step = 1
weight_decay = 0
lr = 5e-5
adam_epsilon = 1e-8
warmup_steps = 0
max_grad_norm = 1.0
kobert_trainer.train(batch_size, max_steps, epochs, acc_step, weight_decay, lr, adam_epsilon, warmup_steps, max_grad_norm)