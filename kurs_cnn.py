import os
from os import listdir
import random
import math
import torch
#import torchvision
from torchvision import transforms
from PIL import ImageFont, ImageDraw, Image
import torch.optim #as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
transforms = transforms.Compose([transforms.Resize(256),
                                 transforms.CenterCrop(256),
                                 transforms.ToTensor(),
                                 normalize])
#TARGET: [isCat, isDog]
train_data_list = []
target_list = []
train_data = []
waited = False
files = listdir('train/')
for i in range(len(listdir('train/'))):
    if len(train_data) == 58 and not waited:
        waited = True
        continue
    f = random.choice(files)
    files.remove(f)
    img = Image.open("train/" + f)
    img_tensor = transforms(img) # (3,256,256)
    train_data_list.append(img_tensor)
    isCat = 1 if 'cat' in f else 0
    isDog = 1 if 'dog' in f else 0
    target = [isCat, isDog]
    target_list.append(target)
    if len(train_data_list) >= 64:
        train_data.append((torch.stack(train_data_list), target_list))
        train_data_list = []
        target_list = []
        print('Loaded batch ', len(train_data), 'of ', int(len(listdir('train/'))/64))
        print('Percentage Done: ', 100*len(train_data)/int(len(listdir('train/'))/64), '%')
        if len(train_data) > 150:
            break


class Netz(nn.Module):
    def __init__(self):
        super(Netz, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 12, kernel_size=5)
        self.conv3 = nn.Conv2d(12, 18, kernel_size=5)
        self.conv4 = nn.Conv2d(18, 24, kernel_size=5)
        self.fc1 = nn.Linear(3456, 1000)
        self.fc2 = nn.Linear(1000, 2)
        self.drop = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.drop(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.drop(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.drop(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = x.view(-1, 3456)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return F.sigmoid(x)


model = Netz()


optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)



def train(epoch):
    model.train()
    batch_id = 0
    for data, target in train_data:

        target = torch.Tensor(target)
        data = Variable(data)
        target = Variable(target)
        optimizer.zero_grad()
        out = model(data)
        criterion = F.mse_loss #binary_cross_entropy
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_id * len(data), len(train_data),
            100. * batch_id / len(train_data), loss.data[0]))
        batch_id = batch_id + 1


def test():
    model.eval()
    files = listdir('test/')
    f = random.choice(files)
    img = Image.open('test/' + f)
    img_eval_tensor = transforms(img)
    img_eval_tensor.unsqueeze_(0)
    data = Variable(img_eval_tensor)
    out = model(data)
    #print(out.data.max(1, keepdim=True)[1])
    #img.show()
    #x = input('')
    return out.data.max(1, keepdim=True)[1]




def centered_crop(cimg, new_height, new_width):
    """
    Source: https://stackoverflow.com/questions/16646183/crop-an-image-in-the-centre-using-pil
    """
    width, height = cimg.size   # Get dimensions
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    return cimg.crop((left, top, right, bottom))


def test_single():
    """
    Test the model on a random image
    """
    model.eval()
    test_files = os.listdir('test/')
    ff = random.choice(test_files)
    test_img = Image.open('test/' + ff)
    img_eval_tensor = transforms(test_img)
    img_eval_tensor.unsqueeze_(0) # _ heisst "inplace", also die Veränderung wird nicht nur ausgegeben, sondern auch
        # tatsächlich auf die Daten in img_tensor übernommen. Ergibt: (1,3,256,256)
    if torch.cuda.is_available():
        img_eval_tensor.cuda()
    data = Variable(img_eval_tensor)
    out = model(data)

    #Ergebnis Ausgabe
    test_result = out.data.max(1, keepdim=True)[1]
    #print(test_result)
    if int(test_result) == 0:
        print('This is a cat.')
    else:
        print('This is a doggo.')
    test_img.show()
    #x = input('')


def test_multiple(img_nr):
    """
    Test the model on multiple random images
    """
    model.eval()

    # Make a list of image filenames
    test_files = os.listdir('test_named/')
    image_list = []
    test_target_list = []
    test_predictions_list = []
    for i in range(img_nr):
        rand_img = random.choice(test_files)
        image_list.append(rand_img)
        test_img = Image.open('test_named/' + rand_img)
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
        print("test_result is: ", test_result)

        # Make a list with targets, which has [1, 0] for a cat, and [0, 1] for a dog
        is_cat = 1 if 'cat' in rand_img else 0
        is_dog = 1 if 'dog' in rand_img else 0
        test_target_label = [is_cat, is_dog]
        test_target_list.append(test_target_label)
        test_files.remove(rand_img)

    print("What the CNN predicted:", test_predictions_list)
    print("True Classes:", test_target_list)
    create_collage_test(image_list, test_target_list, test_predictions_list,
                        thumb_size=150, partially_filled_column=False)


def create_collage_test(image_list, target_list, prediction_list, thumb_size=150, partially_filled_column=False):
    """
    Create an image collage with PIL.
    Images are chosen randomly from test-folder.
    Adapted from: https://stackoverflow.com/questions/35438802/making-a-collage-in-pil#35460517

    image_list -> list of test image names to include in the collage
    target_list -> list of true classes of images in image_list
    prediction_list -> list of predicted classes of images in image_list

    This function is best called with square numbers of images in image_list, so the collage looks nice.
    Note that if thumb_size is chosen too high , there appears a "padding" between the images
    because the images aren't really as big as the space, that gets allocated for them in the collage. So keep
    thumb_size fairly small (< ~400).
    """

    img_nr = len(image_list)
    # Find out best number of cols and rows
    a = math.sqrt(img_nr)
    if math.floor(a) == a:
        cols = int(a)
        rows = int(a)
    elif math.floor(a) + 0.5 < a:
        cols = math.floor(a)+1
        rows = math.floor(a)+1
    else:
        cols = math.floor(a)
        rows = math.floor(a)+1
    #print("Rows", rows, "Cols", cols)


    # Create a new PIL image with the right size for all the tumbs + padding
    new_img_width = cols*(thumb_size+2)
    new_img_height = rows * (thumb_size + 2)
    new_im = Image.new('RGB', (new_img_width, new_img_height))

    # Resize images and put them in a list
    ims = []
    for p in image_list:
        im = Image.open('test_named/' + p)
        im = centered_crop(im, 256, 256) # 256px center crop, because neural net also sees images like that
        im.thumbnail((thumb_size, thumb_size), Image.ANTIALIAS)
        ims.append(im)

    font = ImageFont.truetype("MonoSpatial.ttf", 12)
    i = x = y = 0
    if partially_filled_column: # TODO farbkennung auch bei true einbauen, wie unten bei false
        # Put test-images into the new PIL image. columsn first
        for col in range(cols):
            for row in range(rows):
                new_im.paste(ims[i], (x, y))
                # Draw the Classes over the image
                draw = ImageDraw.Draw(new_im)
                if target_list[i][0] == 1:
                    draw.rectangle([x, y, x + 17, y + 14], fill=(50, 50, 50, 255), outline=None)
                    draw.text((x, y), "Cat", font=font)
                else:
                    draw.rectangle([x, y, x + 17, y + 14], fill=(50, 50, 50, 255), outline=None)
                    draw.text((x, y), "Dog", font=font)
                #draw.rectangle([x, y, x + 17, y + 14], fill=(50, 50, 50, 255), outline=None)

                y += thumb_size + 2
                # Check if all imgages are already in the frame, so we don't get error "list index out of range"
                if i+1 < img_nr:
                    i += 1
                else:
                    break
            x += thumb_size + 2
            y = 0
    else:
        # Fill rows first
        for row in range(rows):
            for col in range(cols):
                new_im.paste(ims[i], (y, x))
                # Draw the Classes over the image
                draw = ImageDraw.Draw(new_im)
                if target_list[i][0] == 1: # if it's a cat
                    if int(prediction_list[i]) == 1:
                        draw.rectangle([y, x, y + 17, x + 14], fill=(50, 50, 50, 255), outline=None)
                        draw.text((y, x), "Cat", font=font, fill=(0, 255, 0, 255))
                    elif int(prediction_list[i]) == 0:
                        draw.rectangle([y, x, y + 17, x + 14], fill=(50, 50, 50, 255), outline=None)
                        draw.text((y, x), "Cat", font=font, fill=(255, 0, 0, 255))
                    else:
                        print("Something went terribly wrong!")
                else: # if it's a dog
                    if int(prediction_list[i]) == 0:
                        draw.rectangle([y, x, y + 17, x + 14], fill=(50, 50, 50, 255), outline=None)
                        draw.text((y, x), "Dog", font=font, fill=(0, 255, 0, 255))
                    elif int(prediction_list[i]) == 1:
                        draw.rectangle([y, x, y + 17, x + 14], fill=(50, 50, 50, 255), outline=None)
                        draw.text((y, x), "Dog", font=font, fill=(255, 0, 0, 255))
                    else:
                        print("Something went terribly wrong!")

                y += thumb_size + 2
                # Check if all imgages are already in the frame, so we don't get error "list index out of range"
                if i+1 < img_nr:
                    i += 1
                else:
                    break
            x += thumb_size + 2
            y = 0

    # Save and show final image
    new_im.save("VSCode_Collage.jpg")
    new_im.show()



for epoch in range(1, 3):
    train(epoch)

# ergebnisse = []
# for sdfsf in range(1, 10):
#     erg = test()
#     print("erg is: ", erg)
#     ergebnisse.append(int(erg))
# print(ergebnisse)

test_multiple(36)

