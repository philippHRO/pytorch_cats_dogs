import os
import math
import torch
from torchvision import transforms, utils
#from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as tfunc
import torch.nn as nn
from PIL import ImageFont, ImageDraw, Image # actually uses Pillow not PIL
import random

thisPath = 'PyTorch/3_HundeUndKatzen/'
# define a filename for model saving and loading
myFileName = "netz_catdog_kernel2.pt"

# Bildtransformationen
transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Mean je Kanal
        std=[0.229, 0.224, 0.225])  # Std-Abw. vom Mean je Kanal
    ])


def load_images(break_nr=0):
    """Load Images."""
    training_data_list = []
    target_list = []
    global train_data
    train_data = []
    files = os.listdir(thisPath + 'train/')

    for i in range(len(os.listdir(thisPath + 'train/'))):
        # Bilder zufällig auswählen, damit nicht erst alle katzen und dann alle hunde in das Netz geladen werden
        # denn dann würde das Netz die Reihenfolge der klassen einfach auswändig lernen.
        f = random.choice(files)
        files.remove(f)

        # Bild öffnen und Transformationen anwenden
        img = Image.open(thisPath + 'train/' + f)
        img_tensor = transforms(img) # ergibt: (3, 256, 256)
        training_data_list.append(img_tensor)

        # Erstelle eine Liste mit je [1, 0] für Katzen und [0, 1] für Hunde
        is_cat = 1 if 'cat' in f else 0
        is_dog = 1 if 'dog' in f else 0
        target_label = [is_cat, is_dog]
        target_list.append(target_label)

        # Erstellt eine Liste aus 64er-Batches
        if len(training_data_list) >= 64:
            train_data.append((torch.stack(training_data_list), target_list))
            training_data_list = [] # alte Informationen verwerfen, um für neuen batch frei zu sein.
            target_list = []
            print('Loaded image batch ', len(train_data), 'of ', int(len(os.listdir(thisPath + 'train/'))/64), ' (',
                  format(100*len(train_data)/int(len(os.listdir(thisPath + 'train/'))/64), '.2f'), '%)')
            # if len(train_data) > 15:
            #     break
            # Stop loading after length of train_data reaches break_nr is given.
            if break_nr != 0 and len(train_data) > break_nr:
                break


class Netz(nn.Module):
    def __init__(self):
        super(Netz, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=5)
        self.conv3 = nn.Conv2d(12, 18, kernel_size=5)
        self.conv4 = nn.Conv2d(18, 24, kernel_size=5)
        self.fc1 = nn.Linear(3456, 1000) # 3456 = 24*12*12 (siehe unten)
        self.fc2 = nn.Linear(1000, 2) # runter auf 2 Klassen für Hund und Katzen

    def forward(self, x):
        x = self.conv1(x)
        x = tfunc.max_pool2d(x, 2)
        x = tfunc.relu(x)
        x = self.conv2(x)
        x = tfunc.max_pool2d(x, 2)
        x = tfunc.relu(x)
        x = self.conv3(x)
        x = tfunc.max_pool2d(x, 2)
        x = tfunc.relu(x)
        x = self.conv4(x)
        x = tfunc.max_pool2d(x, 2)
        x = tfunc.relu(x)
        #print(x.size())    # -> kommt raus: torch.Size([64, 24, 12, 12])
                            # d.h. für ein Bild haben wir nun 24 12*12 Bilder
        x = x.view(-1, 3456)
        x = tfunc.relu(self.fc1(x))
        x = self.fc2(x)
        return tfunc.sigmoid(x)


def train(epoch_count):
    """
    Train the model
    """
    model.train()
    batch_id = 1
    for data, target in train_data:
        target = torch.Tensor(target)
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        data = Variable(data)
        target = Variable(target)
        optimizer.zero_grad()
        out = model(data)
        criterion = tfunc.binary_cross_entropy
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch_count, batch_id, len(train_data),
            100. * batch_id / len(train_data), loss.data[0]))
        batch_id += 1
    # Save Model after every epoch
    torch.save(model, thisPath + myFileName)
    print('Saved Model', myFileName, ' to ', thisPath)


def test_multiple(img_nr):
    """
    Test the model on multiple random images
    """
    model.eval()

    # Make a list of image filenames
    test_files = os.listdir(thisPath + 'test_named/')
    image_list = []
    test_target_list = []
    test_predictions_list = []
    for i in range(img_nr):
        rand_img = random.choice(test_files)
        image_list.append(rand_img)
        test_img = Image.open(thisPath + 'test_named/' + rand_img)
        img_eval_tensor = transforms(test_img)
        img_eval_tensor.unsqueeze_(
            0)  # _ heisst "inplace", also die Veränderung wird nicht nur ausgegeben, sondern auch
        # tatsächlich auf die Daten in img_tensor übernommen. Ergibt: (1,3,256,256)
        if torch.cuda.is_available():
            img_eval_tensor.cuda()
        data = Variable(img_eval_tensor)
        out = model(data)
        test_result = out.data.max(1, keepdim=True)[1] # Ergebnis Ausgabe
        test_predictions_list.append(int(test_result))
        print(test_result)

        # Make a list with targets, which has [1, 0] for a cat, and [0, 1] for a dog
        is_cat = 1 if 'cat' in rand_img else 0
        is_dog = 1 if 'dog' in rand_img else 0
        test_target_label = [is_cat, is_dog]
        test_target_list.append(test_target_label)
        test_files.remove(rand_img)

    print("Classes, predicted by CNN:", test_predictions_list)
    print("True Classes fom image filenames:", test_target_list)
    #create_collage_test(image_list, test_target_list, test_predictions_list, thumb_size=150, partially_filled_column=False)


if os.path.isfile(thisPath + myFileName):
    print('Loading Model', myFileName, ' from ', thisPath)
    model = torch.load(thisPath + myFileName)
else:
    model = Netz()
    if torch.cuda.is_available():
        model.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.01)

# What to do:
load_images()
for epoch in range(1, 5):
    train(epoch)
test_multiple(20)
