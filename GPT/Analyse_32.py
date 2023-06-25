import re
import matplotlib.pyplot as plt
import os
import numpy as np

path="log"
files= os.listdir(path)
print(files)
for name in files:
    name=name[:-4]
    filepath="log/"+name+".txt"
    train_loss=[]
    train_acc=[]
    train_time=[]
    with open(filepath, 'r',encoding="utf-8") as f:
        lines = f.readlines()
        for i,line in enumerate(lines):
            match1 = re.search(r"Epoch(\d+)/Train", line[0:12])
            if match1:
                line_split=line.split("--")
                train_loss.append(round(float(line_split[2][4:]), 3))
                train_acc.append(round(float(line_split[3][4:]) ,3))
                train_time.append(round(float(line_split[4][7:-1]),3))
            match2 = re.search(r"Epoch(\d+)/Test", line[0:11])
            if(match2):
                line_split = line.split("--")
                print(name)
                print(line_split)
                print(lines[i+1])
    
    
    fig, ax1 = plt.subplots(figsize=(9, 6))
    
    ax1.set_xlabel('Batch_epoch')
    ax1.set_ylabel('Loss')
    ax1.plot(list(range(len(train_loss))),train_loss,color='blue',label="Loss")
    plt.ylim(-0.1,1)
    plt.legend(loc='upper left')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Acc')
    ax2.plot(list(range(len(train_acc))),train_acc, color='red',label="Acc")
    plt.ylim(40,100)
    plt.legend(loc='upper right')
    plt.title(name+"--------Train:Loss/Acc")
    plt.savefig(name+".png")
    # plt.show()
    if(len(train_time)>3000):
        print("1",name,np.average(train_time)*2)
    else:
        print(name,np.average(train_time))
