import paddle
import transformers
from paddlenlp.datasets import load_dataset
import paddlenlp
import numpy as np
import time
import random

name="gpt2-medium-en"

def convert_example(example, tokenizer, max_seq_length=128):
    text = example["text"]
    label = example["label"]
    encoded_inputs = tokenizer(text, max_seq_len=max_seq_length, pad_to_max_seq_len=True)
    # encoded_inputs = tokenizer(text, max_length=max_seq_length,padding=True)
    input_ids = encoded_inputs["input_ids"]
    # token_type_ids = encoded_inputs["token_type_ids"]

    return input_ids,label

def run(dataloader,epochs,epoch,require_train,require_acc):
    loss,correct,total,global_step=0,0,0,0
    global_total=epochs*len(dataloader)
    if (require_acc):
        model.eval()
    if (require_train):
        print("开始训练")
        model.train()

    for i,batch in enumerate(dataloader):
        start_time=time.time()
        input_ids, labels = batch

        input_ids = np.array(input_ids).transpose((1, 0))
        # token_type_ids = np.array(token_type_ids).transpose((1, 0))

        input_ids = paddle.to_tensor(input_ids)
        # token_type_ids = paddle.to_tensor(token_type_ids)
        labels = paddle.to_tensor(labels)

        # logits = model(input_ids, token_type_ids)
        logits = model(input_ids)
        loss = criterion(logits, labels)
        predictions = paddle.argmax(logits, axis=1)
        correct = correct+paddle.sum(paddle.to_tensor(predictions == labels))
        total = total+len(labels)

        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        end_time=time.time()
        if (require_train):
            if i%1==0:
                print("Epoch"+str(epoch)+"/Train:"+str(i)+"/"+str(len(dataloader))+"--Global:"+str(global_step)+"/"+str(global_total)+
                            "--Loss"+str(loss.item())+"--Acc:"+str((100*(correct / total)).item())+"--Time/s:"+str(end_time-start_time))
        global_step = global_step + 1

    if require_acc:
        return loss.item(),100*(correct / total)

print("读取数据")
train_ds,test_ds= load_dataset("imdb",splits=("train", "test"))

tokenizer = paddlenlp.transformers.GPTTokenizer.from_pretrained(name)
train_ds.map(lambda x: convert_example(x, tokenizer))
test_ds.map(lambda x: convert_example(x, tokenizer))

val_indices = random.sample(range(25000), 5000)
test_tiny_ds = paddle.io.Subset(test_ds, val_indices)

epochs = 3
batch_size = 16
train_loader = paddle.io.DataLoader(train_ds, batch_size=batch_size, shuffle=True,drop_last=True)
test_loader = paddle.io.DataLoader(test_ds, batch_size=batch_size, shuffle=True,drop_last=True)
test_tiny_loader = paddle.io.DataLoader(test_tiny_ds, batch_size=batch_size, shuffle=True,drop_last=True)
"""
print("start")
lens=128
for i in range(len(train_ds)):
    test=train_ds[i][0]
    if(len(test)!=lens):
        train_ds[i][0]=pad_and_truncate_sequence(train_ds[i][0],lens,0)

print("finish")
for i in range(len(train_ds)):
    test=train_ds[i][0]
    if(len(test)!=lens):
        print(len(test))
"""

model = paddlenlp.transformers.GPTForSequenceClassification.from_pretrained(name, num_classes=2)
criterion = paddle.nn.CrossEntropyLoss()
optimizer = paddle.optimizer.AdamW(learning_rate=2e-5, parameters=model.parameters())

for epoch in range(epochs):
    model.train()
    run(train_loader,epochs,epoch,True,False)
    paddle.save(model.state_dict(), name+"_{}".format(epoch))
    loss_test,acc_test=run(test_tiny_loader,1,1,False,True)
    print("Epoch"+str(epoch)+"/Test:"+"--Loss:"+str(loss_test)+"--Acc:"+str(acc_test))
    print("---------------------------------------------------")

# state_dict = paddle.load("ernie_model_epoch0.pdparams")
# model.set_state_dict(state_dict)
# run(test_loader,False,True)

