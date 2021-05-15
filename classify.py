import torch
import torch.nn.functional as F
import torchvision.transforms as tt
from PIL import Image

from model import FoodImageClassifer

stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
valid_tfms = tt.Compose([tt.Resize([224, 224]), tt.ToTensor(), tt.Normalize(*stats)])

model = FoodImageClassifer()
model.load_state_dict(torch.load('food_classifier.pth', map_location=torch.device('cpu')))
model.eval()
with open('classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]


def classify(imgpath):
    img = Image.open(imgpath)
    img_ts = valid_tfms(img)
    batch_t = torch.unsqueeze(img_ts, 0)
    out = model(batch_t)
    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]]
