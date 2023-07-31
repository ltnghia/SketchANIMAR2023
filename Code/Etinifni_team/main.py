import torch
from torch import nn
from torchvision import transforms, datasets, models
import torchvision.utils as vutils
from torchvision.utils import flow_to_image
from torchvision.utils import save_image
from torch.utils.data import random_split
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import cv2
from google.colab.patches import cv2_imshow
from torchvision.models import resnet50, ResNet50_Weights
import shutil
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.cluster import KMeans
from PIL import Image, ImageOps
from torch.autograd import Variable
import pandas as pd
ngpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
torch.manual_seed(2)

df = pd.read_csv('./content/original/mapped_SketchQuery_GT_Train.csv')

train = df.loc[0:231, :]

test = df.loc[232:, :]
test.index = range(len(test))
train.to_csv("./content/mapped_SketchQuery_GT_Train.csv", index = False)
test.to_csv('./content/mapped_SketchQuery_GT_Val.csv', index = False)

df = pd.read_csv('./content/original/mapped_SketchQuery_Train.csv')

train = df.loc[0:57, :]
test = df.loc[58:, :]
test.index = range(len(test))
train.to_csv("./content/mapped_SketchQuery_Train.csv", index = False)
test.to_csv('./content/mapped_SketchQuery_Val.csv', index = False)

def custom_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        img = np.array(img)
        img = img[:, :, 0:3]
        img = Image.fromarray(img)
        return img

def get_contain_csv(path):
    rows = pd.read_csv(path)
    return rows

class SHRECDataset_val(torch.utils.data.Dataset):


    def __init__(self, labels, dir_path_view_folder, dir_sketch, transforms):
        """
        dir_sketch: path to folder contain sketch (.png files)
        """
        self.labels = get_contain_csv(labels)
        self.dir_path_view_folder = dir_path_view_folder
        self.dir_sketch = dir_sketch
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {}
        sketch_name, folder_name = self.labels.loc[idx]

        folder_path = os.path.join(self.dir_path_view_folder, str(folder_name))
        sketch_path = os.path.join(self.dir_sketch, sketch_name + ".jpg")

        obj = datasets.ImageFolder(folder_path, transform=self.transforms, loader = custom_loader)
        obj_loader = torch.utils.data.DataLoader(obj, batch_size=12, shuffle=False)
        obj_res = next(iter(obj_loader))


        res = obj_res[0]
        image = Image.open(sketch_path).convert("RGB")
        image = self.transforms(image)

        item['obj'] = res
        item['image'] = image

        return item


    def __len__(self):
        return len(self.labels)

class SHRECDataset_train(torch.utils.data.Dataset):


    def __init__(self, labels, dir_path_view_folder, dir_sketch, transforms):
        """
        dir_sketch: path to folder contain sketch (.png files)
        """
        self.labels = get_contain_csv(labels)
        self.dir_path_view_folder = dir_path_view_folder
        self.dir_sketch = dir_sketch
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {}
        sketch_name, folder_name = self.labels.loc[idx]

        folder_path = os.path.join(self.dir_path_view_folder, str(folder_name))
        sketch_path = os.path.join(self.dir_sketch, sketch_name)

        obj = datasets.ImageFolder(folder_path, transform=self.transforms, loader = custom_loader)
        obj_loader = torch.utils.data.DataLoader(obj, batch_size=12, shuffle=False)
        obj_res = next(iter(obj_loader))


        res = obj_res[0]
        image = custom_loader(sketch_path)
        image = self.transforms(image)

        item['obj'] = res
        item['image'] = image

        return item


    def __len__(self):
        return len(self.labels)

transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
])

dataset_train = SHRECDataset_val(
    labels = './content/mapped_SketchQuery_GT_Train.csv',
    dir_path_view_folder = './content/original/multiviews_with_subfolders_v3',
    dir_sketch = './content/original/sketch_folder',
    transforms=transform,
)
dataset_val = SHRECDataset_val(
    labels = './content/mapped_SketchQuery_GT_Val.csv',
    dir_path_view_folder = './content/original/multiviews_with_subfolders_v3',
    dir_sketch = './content/original/sketch_folder',
    transforms=transform,
)

datasetloader_train = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=11,
    shuffle=True
)
datasetloader_val = torch.utils.data.DataLoader(
    dataset_val,
    batch_size=12,
    shuffle=False
)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Encoder3D(nn.Module):
    def __init__(self, type_train = "concat", trainable = True):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        for p in self.model.parameters():
            p.requires_grad = trainable
        self.model.fc = Identity()
        self.type_train = type_train
    def forward(self, batch):
        batch_resize = batch.view(-1, 3, 224, 224)
        res = self.model(batch_resize)
        res = res.view(batch.shape[0], 12, -1) #(batch.shape[0], 12, 2048)
        if self.type_train == "concate":
            res = res.view(batch.shape[0], -1)
            return res
        elif self.type_train == "max_pool":
            res = torch.max(res, 1).values
            return res

class Encoder2D(nn.Module):
    def __init__(self, trainable = True):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        for p in self.model.parameters():
            p.requires_grad = trainable
        self.model.fc = Identity()
    def forward(self, X):
        out = self.model(X)
        return out

class ProjectionHead_Gelu(nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout = 0.3):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class ProjectionHead_Relu(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, projection_dim, dropout = 0.3):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, projection_dim)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        out = self.projection(x)
        out = self.relu(out)
        out = self.fc(out)
        out = self.layer_norm(out)
        return out

import torch.nn.functional as F
class ShrecModel(nn.Module):
    def __init__(self, type_train_3D = "concate", type_projection = "gelu", type_loss = "non_CLIP", temperature = 1.0, embedding3D = 2048, embedding2D = 2048, hidden_dim = 512, projection_dim = 256):
        super().__init__()
        self.Encoder3D = Encoder3D(type_train = type_train_3D)
        self.Encoder2D = Encoder2D()
        if type_projection == "gelu":
          if type_train_3D == "concate":
            self.projection3D = ProjectionHead_Gelu(embedding_dim=12*embedding3D, projection_dim=projection_dim)
          elif type_train_3D == "max_pool":
            self.projection3D = ProjectionHead_Gelu(embedding_dim=embedding3D, projection_dim=projection_dim)
          else:
            print("You input wrongly.")
          self.projection2D = ProjectionHead_Gelu(embedding_dim=embedding2D, projection_dim=projection_dim)
        elif type_projection == "relu":
          if type_train_3D == "concate":
            self.projection3D = ProjectionHead_Relu(embedding_dim=12*embedding3D, hidden_dim = hidden_dim, projection_dim=projection_dim)
          elif type_train_3D == "max_pool":
            self.projection3D = ProjectionHead_Relu(embedding_dim=embedding3D, hidden_dim = hidden_dim, projection_dim=projection_dim)
          else:
            print("You input wrongly.")
          self.projection2D = ProjectionHead_Relu(embedding_dim=embedding2D, hidden_dim = hidden_dim, projection_dim=projection_dim)
        elif  type_train_3D == "max_pool" and type_projection == "none":
          print("You don't use projection.")
        else:
          print("You input wrongly.")
          exit()
        self.type_projection = type_projection
        self.temperature = temperature
        self.type_loss = type_loss

    def forward(self, Object3D, Object2D):
        features3D = self.Encoder3D(Object3D)
        features2D = self.Encoder2D(Object2D)
        embeddings3D = features3D / features3D.norm(dim=-1, keepdim=True)
        embeddings2D = features2D / features2D.norm(dim=-1, keepdim=True)
        if self.type_projection != "none":
          embeddings3D = self.projection3D(features3D)
          embeddings2D = self.projection2D(features2D)

        logits = (embeddings2D @ embeddings3D.T) / self.temperature
        similarity_2D = embeddings2D @ embeddings2D.T
        similarity_3D = embeddings3D @ embeddings3D.T
        loss = 0
        if self.type_loss == "non_CLIP":
          targets = F.softmax(
              (similarity_2D + similarity_3D) / (2 * self.temperature), dim=-1
          )
          loss = (cross_entropy(logits, targets, reduction='mean') + cross_entropy(logits.T, targets.T, reduction='mean'))/2
        elif self.type_loss == "CLIP":
          targets = torch.arange(Object2D.shape[0]).to(device)
          loss = (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))/2
        else:
          exit()
        return loss

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def evaluate(model_sketch, dataset_test):
    model_sketch.eval()
    running_loss = []
    with torch.no_grad():
        for data in dataset_test:
            inputs = data["obj"].to(device)
            labels = data["image"].to(device)
            loss = model_sketch(inputs, labels)
            running_loss.append(loss.item())
    model_sketch.train()
    return np.mean(running_loss)


def extract_3D(model):
    linkfolder3D = "./content/original/multiviews_with_subfolders_v3"
    file3D_ID = []
    obj3D = 0
    with torch.no_grad():
        for k, i in enumerate(os.listdir(linkfolder3D)):
            if k == 0:
                folder_path = os.path.join(linkfolder3D, i)
                obj = datasets.ImageFolder(folder_path, transform=transform, loader = custom_loader)
                obj_loader = torch.utils.data.DataLoader(obj, batch_size=12, shuffle=False)
                obj_res = next(iter(obj_loader))
                res = torch.unsqueeze(obj_res[0], 0).to(device)
                obj3D = model.Encoder3D(res)
                if model.type_projection != "none":
                    obj3D = model.projection3D(obj3D)
                else:
                    obj3D = obj3D/obj3D.norm(dim=-1, keepdim=True)
                file3D_ID.append(i)
            else:
                folder_path = os.path.join(linkfolder3D, i)
                obj = datasets.ImageFolder(folder_path, transform=transform, loader = custom_loader)
                obj_loader = torch.utils.data.DataLoader(obj, batch_size=12, shuffle=False)
                obj_res = next(iter(obj_loader))
                res = torch.unsqueeze(obj_res[0], 0).to(device)
                embed = model.Encoder3D(res)
                if model.type_projection != "none":
                    embed = model.projection3D(embed)
                else:
                    embed = embed/embed.norm(dim=-1, keepdim=True)
                obj3D = torch.concat([obj3D, embed], dim = 0)
                file3D_ID.append(i)
    return file3D_ID, obj3D

def find_matches(model, type_projection, embeddings3D, query, file3D_ID):
    image_link = os.path.join("./content/original/sketch_folder", query+".jpg")
    image = image = Image.open(image_link).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    embeddings2D = model.Encoder2D(image)
    if type_projection != "none":
      embeddings2D = model.projection2D(embeddings2D)
    else:
      embeddings2D = embeddings2D/embeddings2D.norm(dim=-1, keepdim=True)
    dot_similarity = embeddings2D @ embeddings3D.T

    # multiplying by 5 to consider that there are 5 captions for a single image
    # so in indices, the first 5 indices point to a single image, the second 5 indices
    # to another one and so on.
    list_dot = dot_similarity.squeeze(0).tolist()
    dict_dot = {}
    for i in range(len(file3D_ID)):
        dict_dot[int(file3D_ID[i])] = list_dot[i]

    dict_dot = dict(sorted(dict_dot.items(), key=lambda item: item[1]))
    res = list(dict_dot.keys())
    res.reverse()
    return res

import pandas as pd

#Ground_truth : .csv file
#model ids is the map : query_id -> array of output models
#query_ids: just the ids of query

def calc_nn_accuracy(ground_truth, model_ids, query_ids):
    nn_correct = 0

    for i, query_id in enumerate(query_ids):
        if model_ids[query_id][0] in ground_truth.loc[ground_truth['Sketch Query ID'] == query_id]['Model ID'].values:
            nn_correct += 1

    nn_accuracy = nn_correct / len(query_ids)

    return nn_accuracy

def calc_p_at_10_sum(ground_truth, model_ids, query_ids):
    p_at_10_sum = 0

    for i, query_id in enumerate(query_ids):
        relevant_models = ground_truth.loc[ground_truth['Sketch Query ID'] == query_id]['Model ID'].values
        top_10_models = model_ids[query_id][:10]
        p_at_10 = len(set(relevant_models).intersection(set(top_10_models))) / len(set(relevant_models))
        p_at_10_sum += p_at_10

    p_at_10 = p_at_10_sum / len(query_ids)

    return p_at_10


def calc_ndcg(ground_truth, model_ids, query_ids):
    ndcg_sum = 0

    for i, query_id in enumerate(query_ids):
        relevant_models = set(ground_truth.loc[ground_truth['Sketch Query ID'] == query_id]['Model ID'].values)
        returned_models = model_ids[query_id]
        relevance_scores = [1 if model_id in relevant_models else 0 for model_id in returned_models]

        dcg = relevance_scores[0]
        for j in range(1, len(relevance_scores)):
            dcg += relevance_scores[j] / np.log2(j + 1)

        ideal_scores = sorted(relevance_scores, reverse=True)
        idcg = ideal_scores[0]
        for j in range(1, len(ideal_scores)):
            idcg += ideal_scores[j] / np.log2(j + 1)

        ndcg = dcg / idcg
        ndcg_sum += ndcg

    ndcg = ndcg_sum / len(query_ids)
    return ndcg


def calc_mAP(ground_truth, model_ids, query_ids):
    ap_sum = 0

    for i, query_id in enumerate(query_ids):
        relevant_models = set(ground_truth.loc[ground_truth['Sketch Query ID'] == query_id]['Model ID'].values)
        returned_models = model_ids[query_id]
        precision_sum = 0
        num_relevant = 0

        for j in range(len(returned_models)):
            if returned_models[j] in relevant_models:
                num_relevant += 1
                precision_sum += num_relevant / (j + 1)

        ap = precision_sum / len(relevant_models)
        ap_sum += ap

    mAP = ap_sum / len(query_ids)
    return mAP


def performance_function(ground_truth, model_ids, query_ids):
    return calc_nn_accuracy(ground_truth, model_ids, query_ids), \
        calc_p_at_10_sum(ground_truth, model_ids, query_ids), \
        calc_ndcg(ground_truth, model_ids, query_ids), \
        calc_mAP(ground_truth, model_ids, query_ids), \


def result_val(model):
  model.eval()
  file3D_ID, obj3D = extract_3D(model)
  output_dict = {}
  df = pd.read_csv("./content/mapped_SketchQuery_Val.csv")
  for query in df["ID"]:
    output = find_matches(model, model.type_projection, obj3D, query, file3D_ID)
    output_dict[query] = output

  ground_truth = pd.read_csv("./content/mapped_SketchQuery_GT_Val.csv")
  model.train()
  return performance_function(ground_truth, output_dict, list(output_dict.keys()))
  #(0.8783783783783784, 0.9459459459459459, 0.9056789111416603, 0.8621315373201669)

def train(datatrain, dataval, type_train_3D = "concate", type_projection = "gelu", type_loss = "non_CLIP", temperature = 1.0, hidden_dim = 512, projection_dim = 256, encoder3D_lr = 0.0001, encoder2D_lr = 0.0001
          , projection3D_lr = 0.000001, projection2D_lr = 0.000001, epochs = 50, epoch_decay = 1):
    model_sketch = ShrecModel(type_train_3D = type_train_3D, type_projection = type_projection, type_loss = type_loss,
                              temperature = temperature, hidden_dim = hidden_dim, projection_dim = projection_dim).to(device)
    checkpoint_store = torch.load("./content/original/weight_sketch_CLIP_new_sketch_v3_multview_v3.pt")
    model_sketch.load_state_dict(checkpoint_store["model_state_dict"])
    model_sketch.train()
    params = [
        {"params": model_sketch.Encoder3D.parameters(), "lr": encoder3D_lr},
        {"params": model_sketch.Encoder2D.parameters(), "lr": encoder2D_lr},
    ]
    if type_projection != "none":
        params = [
            {"params": model_sketch.Encoder3D.parameters(), "lr": encoder3D_lr},
            {"params": model_sketch.Encoder2D.parameters(), "lr": encoder2D_lr},
            {"params": model_sketch.projection3D.parameters(), "lr": projection3D_lr},
            {"params": model_sketch.projection2D.parameters(), "lr": projection2D_lr}
        ]
    optimizer = torch.optim.AdamW(params, weight_decay=1.0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma= 0.85)
    total_loss_train = []
    total_loss_test = []
    best_result = -1
    best_loss = 10
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss_train = []
        for i, data in enumerate(datatrain):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data["obj"].to(device)
            labels = data["image"].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            loss = model_sketch(inputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss_train.append(loss.item())
        if (epoch+1) % epoch_decay == 0:
            scheduler.step()
        train_loss = np.mean(running_loss_train)
        test_loss = evaluate(model_sketch, dataval)
        print("Epoch : {}, Train Loss: {:.4f}  Test Loss: {:.4f}".format(epoch+1, train_loss, test_loss))
        result = result_val(model_sketch)
        print(result)
        #print(result_train(model_sketch))
        total_loss_train.append(train_loss)
        total_loss_test.append(test_loss)
        if best_result == -1:
            model_sketch.eval()
            torch.save({'model_state_dict': model_sketch.state_dict()}, "./content/original/weight_sketch_CLIP_new_sketch_v3_multview_v3_finetuned_from_new.pt")
            model_sketch.train()
            best_result = result
            best_loss = test_loss
        elif result[0] > best_result[0] or (result[0] == best_result[0] and best_loss > test_loss):
            print("---------------------------------------------------------")
            model_sketch.eval()
            torch.save({'model_state_dict': model_sketch.state_dict()}, "./content/original/weight_sketch_CLIP_best_sketch_v3_multview_v3_finetuned_from_new.pt")
            best_result = result
            best_loss = test_loss
            model_sketch.train()
        model_sketch.eval()
        torch.save({'model_state_dict': model_sketch.state_dict()}, "./content/original/weight_sketch_CLIP_new_sketch_v3_multview_v3_finetuned_from_new.pt")
        model_sketch.train()
    return model_sketch, total_loss_train, total_loss_test
    print('Finished Training')

type_train_3D = "max_pool"  # max_pool or concate
type_projection = "gelu" # gelu or relu
type_loss = "non_CLIP"  # non_CLIP or CLIP
hidden_dim = 512
projection_dim = 128
temperature = 1.0

from numpy import arange

model, hist_train, hist_test = train(datasetloader_train, datasetloader_val, type_train_3D = type_train_3D, type_loss = type_loss,
                                     type_projection = type_projection, temperature = temperature, hidden_dim = hidden_dim, projection_dim = projection_dim, epochs = 40)

checkpoint_store = torch.load("./content/original/weight_sketch_CLIP_best_sketch_v3_multview_v3_finetuned_from_new.pt")
model = ShrecModel(type_train_3D = type_train_3D, type_projection = type_projection,
                              temperature = temperature, hidden_dim = hidden_dim, projection_dim = projection_dim).to(device)
model.load_state_dict(checkpoint_store["model_state_dict"])
model.to(device)
model.eval()

mapping = pd.read_csv("./content/original/mapping.txt")
mapping.head()

file3D_ID, obj3D = extract_3D(model)
model.eval()

real_name_3D = []
for des in file3D_ID:
    real_name_3D.append(mapping.loc[mapping['new_name'] == (des+".obj")]['original'].values[0].split(".")[0])

def find_matches_test(model, embeddings3D, query, real_name_3D):
    image_link = os.path.join("./content/original/SketchQuery_Test/SketchQuery_Test", query+".jpg")
    image = image = Image.open(image_link).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    embeddings2D = model.Encoder2D(image)
    if type_projection != "none":
      embeddings2D = model.projection2D(embeddings2D)
    else:
      embeddings2D = embeddings2D/embeddings2D.norm(dim=-1, keepdim=True)
    dot_similarity = embeddings2D @ embeddings3D.T

    # multiplying by 5 to consider that there are 5 captions for a single image
    # so in indices, the first 5 indices point to a single image, the second 5 indices
    # to another one and so on.
    list_dot = dot_similarity.squeeze(0).tolist()
    dict_dot = {}
    for i in range(len(real_name_3D)):
        dict_dot[real_name_3D[i]] = list_dot[i]

    dict_dot = dict(sorted(dict_dot.items(), key=lambda item: item[1]))
    res = list(dict_dot.keys())
    res.reverse()
    return res

output_test = {}
df = pd.read_csv("./content/original/SketchQuery_Test.csv")
for query in df["ID"]:
    output = find_matches_test(model, obj3D, query, real_name_3D)
    output_test[query] = output

out_list = list(output_test.values())
print(out_list)

submission = pd.DataFrame(data = out_list)
submission.index = list(output_test.keys())
submission.to_csv("etinifni__SketchANIMAR2023.csv", header=False)