from flask import Flask, render_template, request, url_for, session, redirect
from werkzeug.utils import secure_filename
import os
import requests
import sys
import cv2
import numpy as np
import torch
from utils.functions import *
from PIL import ImageFont, ImageDraw, Image

sys.path.insert(0, './model')
sys.path.insert(1, './utils')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fontpath = "./font/SCDream4.otf"
font = ImageFont.truetype(fontpath, 15)
lp_detect_model = torch.load('localization_model.pth', map_location=str(device))
lp_detect_model.eval()

model = torch.load('total.pth', map_location=str(device))
model.eval()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'file_upload'  # 이미지 업로드 폴더 설정
app.secret_key = 'secret_key'

# 업로드 페이지 html 렌더링
@app.route('/')
def upload_file():
    return render_template('file_upload.html')

# 파일 업로드 처리
@app.route('/uploader', methods=['GET', 'POST'])
def idx_to_char(label):
    characters = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return characters[label]

def uploader_file():
    if request.method == 'POST':
        f = request.files['file']
        # 이미지 저장경로 지정
        filename = secure_filename(f.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(upload_path)
        image_url = url_for('static', filename=f'file_upload/{filename}')  # 이미지 URL 생성

        ##########
        ##### 모델 작동 부분 추가#####
        # 모델 작동 로직을 구현합니다.
        # 모델이 작동한 후 결과 이미지를 save_path에 저장합니다.
        # result_file = ...
        ##########start##########
        image_file = 'file_upload/' + filename

        image = cv2.imread(image_file)
        image, tensor, h, w = platecropping(image)
        org_img = image.squeeze()
        tensor = tensor.to(device)

        pred = lp_detect_model(tensor).squeeze()
        pred = pred.detach().cpu().numpy()
        x1, y1, x2, y2 = int(w * pred[0]), int(h * pred[1]), int(w * pred[2]), int(h * pred[3])

        bbox = org_img[y1 - 5:y2 + 5, x1 - 5:x2 + 5]
        if bbox is None:
            print('no bbox')
            exit()

        # perspective tranform
        try:
            bbox = find_big_box(bbox)
        except Exception as e:
            print(e)
            exit()

        gray = cv2.cvtColor(bbox, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (250, 60))
        image_y, image_x = gray.shape[:2]
        image = gray.copy()
        final_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        gray_mean = np.mean(gray)
        gray[gray < gray_mean * 0.3] = 0
        gray[gray > gray_mean * 0.7] = 255
        thresh = np.where(gray >= gray_mean, 0, 255).astype(np.uint8)

        _, labels = cv2.connectedComponents(thresh)
        mask = np.zeros(thresh.shape, dtype="uint8")

        for (i, label) in enumerate(np.unique(labels)):
            # If this is the background label, ignore.
            # perspective tranform
            try:
                bbox = find_big_box(bbox)
            except Exception as e:
                print(e)
                exit()

            gray = cv2.cvtColor(bbox, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (250, 60))
            image_y, image_x = gray.shape[:2]
            image = gray.copy()
            final_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

            gray_mean = np.mean(gray)
            gray[gray < gray_mean * 0.3] = 0
            gray[gray > gray_mean * 0.7] = 255
            thresh = np.where(gray >= gray_mean, 0, 255).astype(np.uint8)

            _, labels = cv2.connectedComponents(thresh)
            mask = np.zeros(thresh.shape, dtype="uint8")

            for (i, label) in enumerate(np.unique(labels)):
                # If this is the background label, ignore
                if label == 0:
                    continue

                # Otherwise, construct the label mask and count the number of pixels
                labelMask = np.zeros(thresh.shape, dtype="uint8")
                labelMask[labels == label] = 255
                numPixels = cv2.countNonZero(labelMask)

                # If the number of pixels in the component is sufficiently large, then add it to our mask of "large blobs"
                if numPixels > 300:
                    mask = cv2.add(mask, labelMask)

            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            plate_characters = []

            for contour in contours:
                (x, y, w, h) = cv2.boundingRect(contour)
                ratio = h / float(w)
                if 1.5 <= ratio <= 3.5:
                    if h / float(image_y) >= 0.5:
                        cv2.rectangle(final_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        curr_num = gray[y:y + h, x:x + w]
                        curr_num = cv2.resize(curr_num, (28, 28))
                        curr_num = np.expand_dims(curr_num, axis=-1)
                        curr_num = np.expand_dims(curr_num, axis=0)
                        plate_characters.append((x, curr_num))

            # Sort the detected characters from left to right
            plate_characters = sorted(plate_characters, key=lambda x: x[0])

            plate_text = ''
            for _, curr_num in plate_characters:
                curr_num = torch.Tensor(curr_num).to(device)
                outputs = model(curr_num)
                _, predicted = torch.max(outputs.data, 1)
                plate_text += idx_to_char(predicted.item())

            # Create a PIL image for drawing text on the original image
            pil_image = Image.fromarray(org_img)
            draw = ImageDraw.Draw(pil_image)
            draw.text((10, 10), plate_text, font=font, fill=(255, 0, 0))

            # Save the resulting image
            save_path = f'static/result_images/{filename}'
            pil_image.save(save_path)

            # 결과 이미지 URL 생성
            result_image_url = url_for('static', filename=f'result_images/{filename}')
            ##########end##########

            return render_template('result.html', image_url=image_url, result_image_url=result_image_url)

if __name__ == '__main__':
    app.run(debug=True, port=9999)