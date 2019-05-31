import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
root="./src_data/"
epoch_num = 50


class CustomDataset(Dataset):
    def __init__(self, label_file_path, stage='Train'):
        with open(label_file_path, 'r') as f:
            # (image_path(str), image_label(str))
            self.imgs = list(map(lambda line: line.strip().split(' '), f))
        self.stage = stage

    def __getitem__(self, index):
        path, label, _ = self.imgs[index]
        img = transforms.Compose([transforms.Scale(224),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(), ])(Image.open('./src_data/' + self.stage + '/' + path).convert('RGB'))
        label = int(label)
        return img, label

    def __len__(self):
        return len(self.imgs)


train_dataset = CustomDataset('./src_data/Train/Label.TXT', 'Train')
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_dataset = CustomDataset('./src_data/Test/Label.TXT', 'Test')
test_loader = DataLoader(test_dataset, batch_size=10)

#-----------------create the Net and training------------------------
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(50176, 20),#64 * 3 * 3, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 3)
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        res = conv3_out.view(conv3_out.size(0), -1)
        out = self.dense(res)
        return out


model = Net()
print(model)

optimizer = torch.optim.Adam(model.parameters())
loss_func = torch.nn.CrossEntropyLoss()

for epoch in range(epoch_num):
    print('epoch {}'.format(epoch + 1))
    # training-----------------------------
    train_loss = 0.
    train_acc = 0.
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = Variable(batch_x), Variable(batch_y)
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        train_loss += loss.item()
        pred = torch.max(out, 1)[1]
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(train_dataset)), train_acc / (len(train_dataset))))

    # evaluation--------------------------------
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = Variable(batch_x, volatile=True), Variable(batch_y, volatile=True)
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        eval_loss += loss.item()
        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum()
        eval_acc += num_correct.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_dataset)), eval_acc / (len(test_dataset))))
