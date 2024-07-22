import fastai.vision.all
import torch
from PIL import Image
import cv2
model = fastai.vision.all.load_learner(r"D:\machine_learning_AI_Builders\บท4\Classification\Dog_Breed_Classification_with_fastAI\model\model.pkl")
flie_name = r"C:\Users\LEGION\OneDrive\Pictures\test.jpg"

predict = model.predict(flie_name)

img = cv2.imread(filename=flie_name)
if img is None:
    print(f"Error: Unable to load image from {flie_name}")
else:
    # ตัวอย่างการปรับขนาดภาพ
    resized_img = cv2.resize(img,(500,500))

    # ตรวจสอบว่าภาพหลังการปรับขนาดถูกต้องหรือไม่
    if resized_img.shape[0] > 0 and resized_img.shape[1] > 0:
        cv2.putText(resized_img,f"Breed :{predict[0]}",(10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),4)
        cv2.imshow("Resized Image", resized_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error: Resized image has invalid dimensions")