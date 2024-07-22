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
img_df["id"] = img_df.apply(lambda x : os.path.basename(x.path).replace(".jpg",""),axis=1)

print(img_df)