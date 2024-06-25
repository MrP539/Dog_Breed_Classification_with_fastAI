import os  # ใช้สำหรับการจัดการระบบไฟล์และไดเรกทอรี
import shutil  # ใช้สำหรับการจัดการไฟล์และไดเรกทอรีอย่างละเอียด
from glob import glob  # ใช้สำหรับการดึงรายการไฟล์จากการใช้ wildcard ในระบบไฟล์

import fastai.vision.all
import pandas as pd  # ใช้สำหรับการจัดการข้อมูลแบบตาราง (DataFrame)
import numpy as np  # ใช้สำหรับการทำงานกับข้อมูลที่มีมิติ (arrays) และคำนวณทางวิทยาศาสตร์
import matplotlib.pyplot as plt  # ใช้สำหรับการสร้างกราฟและการแสดงผลแผนภูมิ
from tqdm.auto import tqdm  # ใช้สำหรับการแสดงแถบความคืบหน้าในการประมวลผล
from PIL import Image
import torch
# from torchvision import datasets,models, transforms  # ใช้สำหรับโมเดลที่เกี่ยวข้องกับ computer vision และการทำ preprocessing ของภาพ
# from torch.utils.data import DataLoader  # ใช้สำหรับโหลดข้อมูลเข้าสู่โมเดล PyTorch ในรูปแบบ batch และการจัดการกับข้อมูล
# from sklearn.model_selection import train_test_split  # ใช้สำหรับการแบ่งข้อมูลออกเป็นชุด train และ test ในการฝึกสอนโมเดล

img_df = pd.DataFrame(glob("data/train/*.jpg"),columns=["path"]) # สร้าง DataFrame: สร้าง DataFrame img_df โดยใช้ข้อมูลจาก glob() และกำหนดชื่อคอลัมน์ "path" สำหรับที่อยู่ของไฟล์ภาพ

img_df["id"] = img_df["path"].map(lambda x : os.path.basename(x).replace(".jpg",""))#map(function, iterable)// สร้างคอลัมน์ "id": ใช้ map() พร้อมกับ lambda function เพื่อสร้างคอลัมน์ "id" จากชื่อไฟล์ภาพที่ดึงมาจาก op.basename(x) และนำออก ".jpg" ด้วย .replace(".jpg", "")

label_df = pd.read_csv(r"D:\machine_learning_AI_Builders\บท4\Classification\Dog_Breed_Classification\data\labels.csv")
img_df = img_df.merge(label_df,on="id")

######################################################################### Create Data Set ##########################################################################################################
loot_dir = r"data\train_breed"

for _,r in img_df.iterrows(): #iterrows() ใน Pandas ใช้สำหรับการวนลูปผ่านแต่ละแถวใน DataFrame โดยที่แต่ละแถวจะถูกส่งออกเป็น tuple ที่ประกอบด้วย index ของแถวและ Series ของข้อมูลในแถวนั้น ๆ 
                              #ซึ่งจะเป็นชุดข้อมูลของแต่ละคอลัมน์ของ DataFrame นั้น ๆ
    if not os.path.exists(f"{loot_dir}/{r.breed}"):
        os.makedirs(f"{loot_dir}/{r.breed}")
    shutil.copy(r["path"],f"{loot_dir}/{r.breed}")

#สร้าง fields เพื่อกำหนด

fields = fastai.vision.all.DataBlock(                                       #สิ่งที่ต้องกำหนดคือ Datablock
    blocks=(fastai.vision.all.ImageBlock,fastai.vision.all.CategoryBlock),   #blocks ใช้สำหรับใส่คู่ระหว่าง input และ output ในที่นี้คือ ImageBlock, CategoryBlock เป็น task classification
    get_items = fastai.vision.all.get_image_files ,                           #ใช้ get_items ในการดึงภาพออกมาจากโฟลเดอร์ โดย get_image_files เป็นฟังก์ชันของ FastAI ที่ใช้หา path ภาพในโฟลเดอร์ที่กำหนด
    get_y = fastai.vision.all.parent_label ,                                    #ใช้ get_y ในการหาว่า label ของแต่ละภาพคืออะไรโดยจะไปดึงตัว parent dir ของ path นั้นมา ในที่นี้คือใช้ฟังก์ชัน parent_label ของ FastAI  
                                                                                #รูปแบบของการวางโฟลเดอร์เป็นดังนี้ <grant_parent>/<parent>/<img.jpg>
    splitter = fastai.vision.all.RandomSplitter(valid_pct=0.2,seed=42),         #หากมี val_set อยู่แล้วใช้ GrandparentSplitter(valid_name = <val_set>)
    item_tfms=fastai.vision.all.RandomResizedCrop(224,min_scale=0.5),           #item_tfms เป็นการ transform ใน CPU ก่อนที่จะส่งไปยัง GPU ในที่นี่เราจะทำการย่อภาพก่อนที่จะส่งไปยัง GPU
    batch_tfms=fastai.vision.all.aug_transforms()                               #aug_transforms ซึ่งเป็นฟังก์ชันของ FastAI ในการทำให้ภาพมีความหลากหลายมากยิ่งขึ้น
)
########################################################################## Create DataLoders   #######################################################################################################


dls = fields.dataloaders(loot_dir,device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),num_workers = 0, bs=64) #num_workers เป็น 0 เมื่อสร้าง DataLoader เพื่อหลีกเลี่ยงปัญหาที่เกิดจาก multiprocessing ใน Windows

######################################################################### Traing ######################################################################################################################

csv_logger = fastai.vision.all.CSVLogger("result.csv")
learner = fastai.vision.all. vision_learner(dls,fastai.vision.all.resnet34,metrics=[fastai.vision.all.error_rate,fastai.vision.all.accuracy],cbs=csv_logger)
# prine(learner.lr_find()) หาค่า learning_rate ที่เหมาะสม
# print(lr)
learner.fine_tune(epochs=100,freeze_epochs=1,base_lr = 3e-3)
learner.remove_cb(csv_logger)
learner.export("Classifying_breeds.pkl")

result_csv_path = learner.path/"result.csv"
df = pd.read_csv(result_csv_path )

df.columns = df.columns.str.strip()

plt.figure(figsize=(6,6))

plt.plot(df["epoch"],df["train_loss"],label="train_loss",color = "blue",marker="x")
plt.plot(df["epoch"],df["valid_loss"],label="val_loss",color = "red",marker="x")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title('Train and Validation Loss by Epoch')
plt.show()

plt.figure(figsize=(6,6))
plt.plot(df.epoch,df.error_rate,label = "error_rate", color = "blue",marker="x")
plt.xlabel("Epoch")
plt.ylabel("error_rate")
plt.title('Validation error_rate by Epoch')
plt.show()

plt.figure(figsize=(6,6))
plt.plot(df.epoch,df.accuracy,label = "accuracy", color = "blue",marker="x")
plt.xlabel("Epoch")
plt.ylabel("accuracy")
plt.title('Validation accuracy by Epoch')
plt.show()





