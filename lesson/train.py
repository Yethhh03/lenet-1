import torch

from lenet import MyLeNet5
from torch import nn

from torch.optim import lr_scheduler  #导入优化器调度器，用于动态调整学习率。
from torchvision import datasets,transforms  #从 torchvision 库中导入数据集和数据变换操作，用于加载和处理图像数据。
import os

#数据预处理
data_transform = transforms.Compose([
    transforms.ToTensor()
])

# 加载训练数据集，创建一个用于训练的数据加载器，将数据集分成大小为 16 的批次，并打乱数据顺序。
train_dataset = datasets.MNIST(root='./data',train=True,transform=data_transform,download=True)
train_dataloader= torch.utils.data.DataLoader(dataset=train_dataset,batch_size=16,shuffle=True)

# 加载测试数据集
test_dataset = datasets.MNIST(root='./data',train=False,transform=data_transform,download=True)
test_dataloader= torch.utils.data.DataLoader(dataset=test_dataset,batch_size=16,shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# 创建一个 LeNet5 模型对象，并将其移动到之前选择的设备上。
model = MyLeNet5().to(device)

loss_fn = nn.CrossEntropyLoss()

#使用随机梯度下降（SGD）更新模型参数。
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

# 创建一个学习率调度器，每经过 10 个 epoch，学习率会乘以 0.1。
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 它接受四个参数：dataloader（用于加载训练数据的数据加载器）、model（要训练的模型）、loss_fn（损失函数）和 optimizer（优化器）。
def train(dataloader, model, loss_fn, optimizer) :
    loss,current,n=0.0,0.0,0
    for batch,(X,y) in enumerate(dataloader):
        X,y=X.to(device),y.to(device)
        output = model(X)
        cur_loss=loss_fn(output,y)
        _,pred=torch.max(output,axis=1)

        #计算当前批次的准确率cur_acc，通过统计真实标签y与预测值pred相等的数量，并除以批次大小得到准确率。
        cur_acc=torch.sum(y == pred)/output.shape[0]

        #清除参数梯度，以便下一次梯度更新
        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()

        loss += cur_loss.item()
        current +=cur_acc.item()
        n = n+1
    print("train_loss"+str(loss / n))
    print("train_loss" + str(current / n))

def val(dataloader,model,loss_fn):
    model.eval()
    loss, current, n =0.0,0.0,0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            output = model(X)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1
        print("val_loss" + str(loss / n))
        print("val_loss" + str(current / n))

        return current/n

epoch =50
min_acc = 0
for t in range(epoch):
    print(f'epoch{t+1}\n----------------')
    train(train_dataloader,model,loss_fn,optimizer)
    a = val(test_dataloader,model,loss_fn)

    if a > min_acc:
        folder ='save_model'
        if not os.path.exists(folder):
            os.mkdir('save_model')
        min_acc = a
        print('save best model')
        torch.save(model.state_dict(),'save_model/best_model.pth')
print('Done!')






