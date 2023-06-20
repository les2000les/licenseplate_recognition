from flask import Flask, request
import sys
import requests
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

@app.route('/upload', methods=['POST'])
def upload_image():
    # 이미지 파일 저장 경로
    save_path = 'file_upload/getimage.jpg'

    # 전송된 이미지 데이터 받기
    image_file = request.files['image']

    # 이미지 파일 저장
    image_file.save(save_path)
    print(image_file)
    print('이미지 업로드 완료')

    ##########
    ##### 모델 작동 부분 추가#####
    # 모델 작동 로직을 구현합니다.
    # 모델이 작동한 후 결과 이미지를 save_path에 저장합니다.
    # result_file = ...
    ##########start##########
    image = str(image_file.filename)
    print(image)
    image, tensor, h, w = platecropping(image)
    org_img = image.squeeze()
    tensor = tensor.to(device)


    pred = lp_detect_model(tensor).squeeze()
    pred = pred.detach().cpu().numpy()
    x1, y1, x2, y2 = int(w*pred[0]), int(h*pred[1]), int(w*pred[2]), int(h*pred[3])


    bbox = org_img[y1-5:y2+5, x1-5:x2+5]
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
    gray = cv2.resize(gray, (250,60))
    image_y, image_x= gray.shape[:2]
    image =gray.copy()
    final_image =cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    gray_mean=np.mean(gray)
    gray[gray< gray_mean*0.3] =0
    gray[gray> gray_mean*0.7] =255
    thresh = np.where(gray>=gray_mean,0,255).astype(np.uint8)

    _, labels = cv2.connectedComponents(thresh)
    mask = np.zeros(thresh.shape, dtype="uint8")

    for (i, label) in enumerate(np.unique(labels)):
        # If this is the background label, ignore it
        if label == 0:
            continue

        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)

        if  numPixels>60 and numPixels <1400:
            mask = cv2.add(mask, labelMask)

    (contours, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort the contours by their x-axis position, ensuring that we read the numbers from left to right
    contours = sorted([(c, cv2.boundingRect(c)[0]) for c in contours], key=lambda x: x[1])

    ####test
    contour_w = list()
    pre_x, pre_y, pre_w, pre_h = 0,0,0,0
    final_contour=list()
    for (contour, _) in contours:
        # Compute the bounding box for the rectangle
        (x, y, w, h) = cv2.boundingRect(contour)
        if h >3  and w/h<5  and x+w<=image_x: #and h/w<6 마지막으로 변경
            if (x<12 or x>230) and (x+w>=image_x or y+h >=image_y): continue
            if (x<15 or x>180) and (abs(w-h)<=8 or h<15) and w+h<52: continue #x+w>=image_x-1 or
            if (x<15  and w<16 and h<35): continue #and abs(w-h)<=10
            if (w>39): continue
            if (x>200 and w<15 and h<35): continue
            if (x<12 and y+h>=image_y): continue
            if (x>210 and h/w>5): continue
            if (y<5 and abs(w-h)<2): continue
            if (x>210 and y<5 and x+w>=image_x): continue
            if (x>210 and x+w>=image_x and y+h >=image_y): continue
            if ((x< 10 or x >220) and (h <=10 or y+h == image_y)): continue
            if (x <5 and h>50): continue
            if (x <5 and h/w>6): continue
            if (x <5 and abs(w-h)<10): continue
            if (x <5 and h+w<51): continue
            if (x <20 and h+w<45): continue
            if overlap(pre_x,pre_w, x,w):
                x,y,w,h = combination(pre_x,pre_y,pre_w,pre_h, x,y,w,h)
                del contour_w[-1]
                del final_contour[-1]
            contour_w.append(w)
            final_contour.append([x,y,w,h])
            pre_x, pre_y, pre_w, pre_h = x,y,w,h
    w_mean = np.mean(contour_w)

    kor =list()
    final_bbox =list()
    final_char=list()

    if len(final_contour) not in [8,9]:
        for idx,contour in enumerate(final_contour):
            (x, y, w, h) = contour
            # Crop the ROI and then threshold the greyscale ROI to reveal the digit
            roi = image[y:y + h, x:x + w]
            thresh = roi.copy()
            digit=total_detection(thresh, model)
            final_bbox.append([x,y,w,h])
            final_char.append(digit)

    elif len(final_contour) == 8:
        previous_number=0
        previous_sum=0
        for idx,contour in enumerate(final_contour):
            (x, y, w, h) = contour
            if idx not in [2,3]:
                if idx ==7 and w >= w_mean*1.3:
                    w = int(w*0.8)
                roi = image[y:y + h, x:x + w]
                thresh = roi.copy()

                digit=total_detection(thresh, model)
                final_bbox.append([x,y,w,h])
                final_char.append(digit)

                if idx==1:
                    previous_sum = (w ** 2 + h ** 2) ** 0.5
            else :
                if (w ** 2 + h ** 2) ** 0.5 > previous_sum * 0.9 and previous_number==0:
                    roi = image[y:y + h, x:x + w]
                    thresh = roi.copy()
                    digit=total_detection(thresh, model)
                    final_bbox.append([x,y,w,h])
                    final_char.append(digit)
                    previous_number=1
                else:
                    kor.append([x,y,w,h])

        if len(kor) ==1:
            (x,y,w,h) = kor[0]
            if(y>0): y=y-1
            roi = image[y:y + h, x:x + w+1]
            thresh = roi.copy()
            digit=total_detection(thresh, model)
            final_bbox.append([x,y,w,h])
            final_char.append(digit)
            kor.clear()

    elif len(final_contour) == 9:
        for idx,contour in enumerate(final_contour):
            (x, y, w, h) = contour

            if idx not in [3,4]:
                if idx ==8 and w >= w_mean*1.3:
                    w = int(w*0.8)
                roi = image[y:y + h, x:x + w]
                thresh = roi.copy()
                # print(f"Prediction: , w:{w}, h:{h}")
                digit=total_detection(thresh, model)
                final_bbox.append([x,y,w,h])
                final_char.append(digit)
            else:
                kor.append([x,y,w,h])

    if len(kor) >1:
        if (abs(kor[0][0] - kor[1][0])<=10 and kor[0][1] > kor[1][1]):   #(abs(kor[0][0] - kor[1][0])<=5 and kor[0][1] > kor[1][1]):
            y=kor[1][1]
            y_h=kor[0][1]+kor[0][3]
            x=kor[1][0]
            x_w=kor[0][0]+kor[0][2]
        else:
            y=kor[0][1]
            y_h=kor[1][1]+kor[1][3]
            x=kor[0][0]
            x_w=kor[1][0]+kor[1][2]
        h= y_h -y
        w = x_w-x
        if(y>0): y=y-1
        if (x+w<249): w+=1
        if w > 60:
            print('wrong data')
            exit()
        roi = image[y:y + h, x:x + w]
        thresh = roi.copy()
        digit=total_detection(thresh, model)
        final_bbox.append([x,y,w,h])
        final_char.append(digit)

    im_pil = Image.fromarray(final_image)
    draw = ImageDraw.Draw(im_pil)
    for i in range(len(final_bbox)):
        box_location = final_bbox[i]
        x,y,w,h = box_location
        draw.rectangle(xy=(x,y,x+w,y+h), outline=(0,255,0))
        draw.text(xy=(x+5,y+5), text=final_char[i], fill='red', font= font)

    #im_pil.save(f"./result.png","PNG") ## 파일을 저장하는 곳
    ##########

    # 결과 이미지 파일 경로
    result_path = 'file_upload/result.jpg'
    # 결과 이미지를 저장합니다.
    im_pil.save(result_path,"PNG")
    print('결과 이미지 저장 완료')

    # 결과 이미지 파일 열기
    with open(result_path, 'rb') as file:
        # 파일 데이터 읽기
        file_data = file.read()

    # POST 요청으로 이미지 데이터 전송
    response = requests.post('http://127.0.0.1:9999/result', files={'image': file_data})

    # 응답 확인
    if response.status_code == 200:
        print('결과 이미지 전송 성공')
    else:
        print(response.status_code)
        print('결과 이미지 전송 실패')

    return 'Image upload successful'


if __name__ == '__main__':
    app.run()
