from flask import Flask, render_template, request, url_for, session, redirect
from werkzeug.utils import secure_filename
import os
import stat
import requests
import sys
import cv2
import numpy as np
import torch
# from utils.functions import *
from utils.functions import platecropping, total_detection, find_big_box, overlap, combination
from PIL import ImageFont, ImageDraw, Image
import time

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
app.config['UPLOAD_FOLDER'] = 'static'  # 이미지 업로드 폴더 설정
app.secret_key = 'secret_key'

# 업로드 페이지 html 렌더링
@app.route('/')
def upload_file():
    return render_template('file_upload.html')

# 파일 업로드 처리
@app.route('/uploader', methods=['GET', 'POST'])
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
        image_file = 'file_upload/getimage.jpg'

        image = filename
        print(image)
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
            # If this is the background label, ignore it
            if label == 0:
                continue

            labelMask = np.zeros(thresh.shape, dtype="uint8")
            labelMask[labels == label] = 255
            numPixels = cv2.countNonZero(labelMask)

            if numPixels > 60 and numPixels < 1400:
                mask = cv2.add(mask, labelMask)

        (contours, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort the contours by their x-axis position, ensuring that we read the numbers from left to right
        contours = sorted([(c, cv2.boundingRect(c)[0]) for c in contours], key=lambda x: x[1])

        ####test
        contour_w = list()
        pre_x, pre_y, pre_w, pre_h = 0, 0, 0, 0
        final_contour = list()
        for (contour, _) in contours:
            # Compute the bounding box for the rectangle
            (x, y, w, h) = cv2.boundingRect(contour)
            if h > 3 and w / h < 5 and x + w <= image_x:  # and h/w<6 마지막으로 변경
                if (x < 12 or x > 230) and (x + w >= image_x or y + h >= image_y): continue
                if (x < 15 or x > 180) and (abs(w - h) <= 8 or h < 15) and w + h < 52: continue  # x+w>=image_x-1 or
                if (x < 15 and w < 16 and h < 35): continue  # and abs(w-h)<=10
                if (w > 39): continue
                if (x > 200 and w < 15 and h < 35): continue
                if (x < 12 and y + h >= image_y): continue
                if (x > 210 and h / w > 5): continue
                if (y < 5 and abs(w - h) < 2): continue
                if (x > 210 and y < 5 and x + w >= image_x): continue
                if (x > 210 and x + w >= image_x and y + h >= image_y): continue
                if ((x < 10 or x > 220) and (h <= 10 or y + h == image_y)): continue
                if (x < 5 and h > 50): continue
                if (x < 5 and h / w > 6): continue
                if (x < 5 and abs(w - h) < 10): continue
                if (x < 5 and h + w < 51): continue
                if (x < 20 and h + w < 45): continue
                if overlap(pre_x, pre_w, x, w):
                    x, y, w, h = combination(pre_x, pre_y, pre_w, pre_h, x, y, w, h)
                    del contour_w[-1]
                    del final_contour[-1]
                contour_w.append(w)
                final_contour.append([x, y, w, h])
                pre_x, pre_y, pre_w, pre_h = x, y, w, h
        w_mean = np.mean(contour_w)

        kor = list()
        final_bbox = list()
        final_char = list()

        if len(final_contour) not in [8, 9]:
            for idx, contour in enumerate(final_contour):
                (x, y, w, h) = contour
                # Crop the ROI and then threshold the greyscale ROI to reveal the digit
                roi = image[y:y + h, x:x + w]
                thresh = roi.copy()
                digit = total_detection(thresh, model)
                final_bbox.append([x, y, w, h])
                final_char.append(digit)

        elif len(final_contour) == 8:
            previous_number = 0
            previous_sum = 0
            for idx, contour in enumerate(final_contour):
                (x, y, w, h) = contour
                if idx not in [2, 3]:
                    if idx == 7 and w >= w_mean * 1.3:
                        w = int(w * 0.8)
                    roi = image[y:y + h, x:x + w]
                    thresh = roi.copy()

                    digit = total_detection(thresh, model)
                    final_bbox.append([x, y, w, h])
                    final_char.append(digit)

                    if idx == 1:
                        previous_sum = (w ** 2 + h ** 2) ** 0.5
                else:
                    if (w ** 2 + h ** 2) ** 0.5 > previous_sum * 0.9 and previous_number == 0:
                        roi = image[y:y + h, x:x + w]
                        thresh = roi.copy()
                        digit = total_detection(thresh, model)
                        final_bbox.append([x, y, w, h])
                        final_char.append(digit)
                        previous_number = 1
                    else:
                        kor.append([x, y, w, h])

            if len(kor) == 1:
                (x, y, w, h) = kor[0]
                if (y > 0): y = y - 1
                roi = image[y:y + h, x:x + w + 1]
                thresh = roi.copy()
                digit = total_detection(thresh, model)
                final_bbox.append([x, y, w, h])
                final_char.append(digit)
                kor.clear()

        elif len(final_contour) == 9:
            for idx, contour in enumerate(final_contour):
                (x, y, w, h) = contour

                if idx not in [3, 4]:
                    if idx == 8 and w >= w_mean * 1.3:
                        w = int(w * 0.8)
                    roi = image[y:y + h, x:x + w]
                    thresh = roi.copy()
                    # print(f"Prediction: , w:{w}, h:{h}")
                    digit = total_detection(thresh, model)
                    final_bbox.append([x, y, w, h])
                    final_char.append(digit)
                else:
                    kor.append([x, y, w, h])

        if len(kor) > 1:
            if (abs(kor[0][0] - kor[1][0]) <= 10 and kor[0][1] > kor[1][
                1]):  # (abs(kor[0][0] - kor[1][0])<=5 and kor[0][1] > kor[1][1]):
                y = kor[1][1]
                y_h = kor[0][1] + kor[0][3]
                x = kor[1][0]
                x_w = kor[0][0] + kor[0][2]
            else:
                y = kor[0][1]
                y_h = kor[1][1] + kor[1][3]
                x = kor[0][0]
                x_w = kor[1][0] + kor[1][2]
            h = y_h - y
            w = x_w - x
            if (y > 0): y = y - 1
            if (x + w < 249): w += 1
            if w > 60:
                print('wrong data')
                exit()
            roi = image[y:y + h, x:x + w]
            thresh = roi.copy()
            digit = total_detection(thresh, model)
            final_bbox.append([x, y, w, h])
            final_char.append(digit)

        im_pil = Image.fromarray(final_image)
        draw = ImageDraw.Draw(im_pil)
        for i in range(len(final_bbox)):
            box_location = final_bbox[i]
            x, y, w, h = box_location
            draw.rectangle(xy=(x, y, x + w, y + h), outline=(0, 255, 0))
            draw.text(xy=(x + 5, y + 5), text=final_char[i], fill='red', font=font)

        # im_pil.save(f"./result.png","PNG") ## 파일을 저장하는 곳
        ##########

        # 결과 이미지 파일 경로
        result_filename = 'result.jpg'
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        # 결과 이미지를 저장합니다.
        im_pil.save(result_path, "JPEG")
        result_text = ''.join(final_char)
        print(result_text)
        print('결과 이미지 저장 완료')
        if os.path.exists(result_path):
            os.chmod(result_path, 0o644)
            print("result.jpg 파일이 생성됨")
            # return redirect('/show_result')
            # return render_template('result.html')
            # 결과 이미지 파일 URL
            filename = 'static/result.jpg'


            # # 파일의 현재 권한
            # current_permissions = stat.S_IMODE(os.lstat(filename).st_mode)
            #
            # # 읽기 및 쓰기 권한을 추가
            # new_permissions = current_permissions | stat.S_IRUSR | stat.S_IWUSR
            #
            # # 파일의 권한을 업데이트합
            # os.chmod(filename, new_permissions)
            # result_image_url = 'file_upload/result.jpg'
            return render_template('result.html', result_image_url = result_path)


        else:
            print("result.jpg 파일이 생성되지 않음")
            return "False"

if __name__ == '__main__':
    app.run(port = 9999, debug=True)