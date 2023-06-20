import json
import os
from pickle import TRUE
import torch
import random
import numpy as np
import cv2
import os
import math
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

korean = {'0':'가', '1':'나', '2':'다', '3':'라', '4':'마', '5':'거', '6':'너', '7':'더', '8':'러', '9':'머', '10':'버', '11':'서',
                   '12':'어', '13':'저', '14':'고', '15':'노', '16':'도', '17':'로', '18':'모', '19':'보', '20':'소', '21':'오', '22':'조', '23':'구', '24':'누',
                     '25':'두', '26':'루', '27':'무', '28':'부', '29':'수', '30':'우', '31':'주', '32':'하', '33':'허', '34':'호'}

char = {'0':'0', '1':'1', '2':'2', '3':'3', '4':'4', '5':'5', '6':'6', '7':'7', '8':'8', '9':'9',
          '10':'가', '11':'나', '12':'다', '13':'라', '14':'마', '15':'거', '16':'너', '17':'더', '18':'러', '19':'머', '20':'버', '21':'서',
                   '22':'어', '23':'저', '24':'고', '25':'노', '26':'도', '27':'로', '28':'모', '29':'보', '30':'소', '31':'오', '32':'조', '33':'구', '34':'누',
                     '35':'두', '36':'루', '37':'무', '38':'부', '39':'수', '40':'우', '41':'주', '42':'하', '43':'허', '44':'호'}

m= torch.nn.Sigmoid()

TARGET_WIDTH = 128
TARGET_HEIGHT = 128 
minus_data =0
transform_image = transforms.Compose(
        [   
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Grayscale(), 
        ]
    )

def create_data_lists(train_path, output_folder):
    """
    Create lists of images, the bounding boxes and labels of the objects in these images, and save these to file.
    :param path_train: path that data exists
    :param output_folder: folder where the JSONs must be saved
    """
    #train data
    data_path = os.path.abspath(train_path)

    train_images = list()
    label = list()
    n_objects =0
    dir_list = os.listdir(data_path)

    for dir in dir_list:
        file_list= os.listdir(os.path.join(data_path, dir))
        for id in file_list:
            train_images.append(os.path.join(data_path,dir,id)) # append image
            label_num = int(dir)
            label.append(label_num)
            n_objects += 1

    assert len(label) == len(train_images)

    # Save to file
    with open(os.path.join(output_folder, 'TRAIN_images.json'), 'w') as j:
        json.dump(train_images, j)
    with open(os.path.join(output_folder, 'TRAIN_objects.json'), 'w') as j:
        json.dump(label, j)

    print('\nThere are %d training images containing a total of %d objects. Files have been saved to %s.' % (
        len(train_images), n_objects, os.path.abspath(output_folder)))


def transform(image):

    new_image = image

    #threshold image
    new_image= np.where(new_image>=np.mean(new_image),0, 255).astype(np.float32)
    new_image = cv2.blur(new_image, (2,2))
    # normalize
    new_image = new_image/255

    return new_image

def createFolder(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(os.path.join(os.getcwd(),dir))
            for i in range(-1,45):
                os.makedirs(os.path.join(os.getcwd(),dir,str(i)))
    except OSError:
        print('Error: Creating directory. ' +  dir)

def platecropping(image):
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    height, width = image.shape[0], image.shape[1]
    tensor = image / 255.
    #tensor = image[np.newaxis, :, :]  # (C, H, W)
    tensor = image[:, :, np.newaxis]  # (H, W, C)
    tensor = transform_image(tensor)
    return image, tensor, height, width

def find_big_box(image):
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    
    # image = FT.resize(image, (224,224))
    image = np.array(image)
    image_RGB = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    thresh = np.where(image>np.mean(image)//20*20, 255, 0).astype(np.uint8)
    thresh = cv2.erode(thresh, erode_kernel)

    _, labels = cv2.connectedComponents(thresh)

    thresh_list =list()
    thresh_idx =list()

    for (i, label) in enumerate(np.unique(labels)):
        if label == 0:
            continue

        # Otherwise, construct the label mask to display only connected component
        # for the current label
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        thresh_idx.append(numPixels)
        thresh_list.append(labelMask)
    if not thresh_idx:
        print("pass")
        return image_RGB
#     print(thresh_idx)
    img =thresh_idx.index(max(thresh_idx))
    img = thresh_list[img]
    row, col = img.shape
    arr = np.empty((0,2), int)

    for r in range(row):
        for c in range(col):
            if(img[r,c] > 0): 
                arr = np.append(arr, np.array([[c,r]]), axis=0)
    
    sm = arr.sum(axis=1)                 # 4쌍의 좌표 각각 x+y 계산
    diff = np.diff(arr, axis = 1)       # 4쌍의 좌표 각각 x-y 계산

    topLeft = arr[np.argmin(sm)]         # x+y가 가장 값이 좌상단 좌표
    bottomRight = arr[np.argmax(sm)]     # x+y가 가장 큰 값이 우하단 좌표
    topRight = arr[np.argmin(diff)]     # x-y가 가장 작은 것이 우상단 좌표
    bottomLeft = arr[np.argmax(diff)]   # x-y가 가장 큰 값이 좌하단 좌표
    
    x1,y1 = topLeft
    x2, y2 = topRight
    x3, y3 = bottomRight
    x4, y4 = bottomLeft
    degree = cal_rad([x1,y1,x2,y2])
    # print(f"degree: {degree}")
    # if(degree >-4  and degree <4.5):
    if(degree >-4  and degree <0.01):
        # if(y1 > y2):
        #     result = image_RGB[y1:y3,x1:x3,:]        
        # else:
        #     result = image_RGB[y2:y4,x2:x4,:]
        result = image_RGB[min(y1,y2):max(y3,y4), min(x1,x4):max(x2,x3),:]
    else:
        result = find_perspective(topLeft, topRight, bottomRight , bottomLeft, image_RGB)
    
    return result

def find_perspective(topLeft, topRight, bottomRight , bottomLeft, image_RGB):
    pts1 = np.float32([topLeft, topRight, bottomRight , bottomLeft])

    # 변환 후 영상에 사용할 서류의 폭과 높이 계산 ---③ 
    w1 = abs(bottomRight[0] - bottomLeft[0])    # 상단 좌우 좌표간의 거리
    w2 = abs(topRight[0] - topLeft[0])          # 하당 좌우 좌표간의 거리
    h1 = abs(topRight[1] - bottomRight[1])      # 우측 상하 좌표간의 거리
    h2 = abs(topLeft[1] - bottomLeft[1])        # 좌측 상하 좌표간의 거리
    width = int(max([w1, w2]))                       # 두 좌우 거리간의 최대값이 서류의 폭
    height = int(max([h1, h2]))                      # 두 상하 거리간의 최대값이 서류의 높이

    # 변환 후 4개 좌표
    pts2 = np.float32([[0,0], [width-1,0], 
                        [width-1,height-1], [0,height-1]])

    # 변환 행렬 계산 
    mtrx = cv2.getPerspectiveTransform(pts1, pts2)
    # 원근 변환 적용
    result = cv2.warpPerspective(image_RGB, mtrx, (width, height))

    return result

def cal_rad(arr):
    rad = math.atan2(arr[3]-arr[1], arr[2]-arr[0])
    result = radTodegree(rad)
    return result
def radTodegree(rad):
    PI = math.pi
    deg = (rad*180)/PI
    return deg

def detection(img, digit_recogn_model):
    # crop = img
    # crop= np.where(crop>=np.mean(crop),255,0).astype(np.float32)
    gray = img
    gray_mean=np.mean(gray)
    gray[gray< gray_mean*0.5] =0
    gray_mean=np.mean(gray)
    gray = cv2.add(gray_mean*0.1,gray)
    crop = np.where(gray>=(gray_mean),255,0).astype(np.uint8)

    # Get width, height for each cropped image
    # and calculate the padding to match the image input of digits recognition model
    width = crop.shape[1]
    height = crop.shape[0]
    padding_width = (TARGET_WIDTH - width) // 2 if width < TARGET_WIDTH else int(0.45 * width)
    padding_width //= 4
    padding_height = (TARGET_HEIGHT - height) // 2 if height < TARGET_HEIGHT else int(0.17 * height)
    padding_height //= 4

    # Apply padding and resize
    crop = cv2.copyMakeBorder(crop, padding_height, padding_height, padding_width, padding_width, cv2.BORDER_CONSTANT, None, 255)
    # crop = cv2.copyMakeBorder(crop, padding_height, padding_height, padding_width, padding_width, cv2.BORDER_CONSTANT)
    crop = cv2.resize(crop, (TARGET_WIDTH, TARGET_HEIGHT))

    # Prepare data for inference
    crop = crop[np.newaxis, np.newaxis, :, :]
    crop = torch.tensor(crop, dtype=torch.float32) / 255.
    crop = crop.to(device)

    # make evaluation
    pred = digit_recogn_model(crop)
    pred = pred.detach().cpu().numpy()
    pred = pred.argmax(1)
    pred = str(pred[0])

    return pred

def kor_detection(img, korean_model):
    # crop = img
    # crop[crop<np.mean(crop)*0.9]=0
    # crop= np.where(crop>=np.mean(crop),0, 255).astype(np.float32)
    gray = img

    gray_mean=np.mean(gray)
    gray[gray< gray_mean*0.5] =0
    gray_mean=np.mean(gray)
    gray = cv2.add(gray_mean*0.11,gray)
    
    crop = np.where(gray>gray_mean,0,255).astype(np.uint8)
    
    # Get width, height for each cropped image
    # and calculate the padding to match the image input of digits recognition model
    width = crop.shape[1]
    height = crop.shape[0]
    padding_width = (64 - width) // 2 
    padding_height = (64 - height) // 2 
    
    # Apply padding and resize
    crop = cv2.copyMakeBorder(crop, padding_height, padding_height, padding_width, padding_width, cv2.BORDER_CONSTANT)
    crop = cv2.resize(crop, (64,64))
    # crop= cv2.blur(crop, (2,2))
    # cv2.imwrite(f"./char.png", crop)
    # exit()
    # Prepare data for inference
    crop = crop[np.newaxis, np.newaxis, :, :]
    crop = torch.tensor(crop, dtype=torch.float32)/ 255.
    crop = crop.to(device)


    # make evaluation
    pred = korean_model(crop)
    pred = pred.detach().cpu().numpy()
    pred = pred.argmax(1)
    pred = str(pred[0])

    # pred = korean[str(pred[0])]

    return pred


def kor_detect(img,digit, num, idx):
    # crop = img
    # crop[crop<np.mean(crop)*0.9]=0
    # crop= np.where(crop>=np.mean(crop),0, 255).astype(np.float32)
    gray = img

    gray_mean=np.mean(gray)
    gray[gray< gray_mean*0.5] =0
    gray_mean=np.mean(gray)
    gray = cv2.add(gray_mean*0.1,gray)
    
    crop = np.where(gray>gray_mean,255,0).astype(np.uint8)
    
    # Get width, height for each cropped image
    # and calculate the padding to match the image input of digits recognition model
    width = crop.shape[1]
    height = crop.shape[0]
    padding_width = (64 - width) // 2 
    padding_height = (64 - height) // 2 
    
    # Apply padding and resize
    crop = cv2.copyMakeBorder(crop, padding_height, padding_height, padding_width, padding_width, cv2.BORDER_CONSTANT, None, 255)
    crop = cv2.resize(crop, (64,64))
    cv2.imwrite(f"./korean/{digit}/{num}_{idx}.png", crop)

def num_detect(img, digit,num, idx):
    # crop = img
    # crop[crop<np.mean(crop)*0.9]=0
    # crop= np.where(crop>=np.mean(crop),0, 255).astype(np.float32)
    gray = img

    gray_mean=np.mean(gray)
    gray[gray< gray_mean*0.5] =0
    gray_mean=np.mean(gray)
    gray = cv2.add(gray_mean*0.1,gray)
    
    try:
        crop = np.where(gray>gray_mean,255,0).astype(np.uint8)
    except:
        print(num)
    # Get width, height for each cropped image
    # and calculate the padding to match the image input of digits recognition model
    width = crop.shape[1]
    height = crop.shape[0]
    padding_width = (64 - width) // 2 
    padding_height = (64 - height) // 2 
    
    # Apply padding and resize
    crop = cv2.copyMakeBorder(crop, padding_height, padding_height, padding_width, padding_width, cv2.BORDER_CONSTANT, None, 255)
    crop = cv2.resize(crop, (64,64))
    cv2.imwrite(f"./number/{digit}/{num}_{idx}.png", crop)

def total_detection(img, model):
    global minus_data
    gray = img

    gray_mean=np.mean(gray)
    gray[gray< gray_mean*0.5] =0
    gray_mean=np.mean(gray)
    gray = cv2.add(gray_mean*0.1,gray)
    
    try:
        crop = np.where(gray>gray_mean,0,255).astype(np.uint8)
    except:
        return -2
    width = crop.shape[1]
    height = crop.shape[0]
    padding_width = (64 - width) // 2 
    padding_height = (64 - height) // 2 
    
    # Apply padding and resize
    crop = cv2.copyMakeBorder(crop, padding_height, padding_height, padding_width, padding_width, cv2.BORDER_CONSTANT)
    crop = cv2.resize(crop, (64,64))
  
    crop = crop[np.newaxis, np.newaxis, :, :]
    crop = torch.tensor(crop, dtype=torch.float32)/ 255.
    crop = crop.to(device)


    # # make evaluation
    # pred = model(crop)
    # # pred = pred.detach().cpu().numpy()
    # pred = m(pred)
    # max_idx=pred.argmax(1)
    # if(pred[0,max_idx]>=0.9):
    #     final = str(max_idx.item())
    # else:
    #     final ='-1'
    #     minus_data +=1
    # # print(final, '\t', pred[0,max_idx].item())

    pred = model(crop)
    pred = pred.detach().cpu().numpy()
    pred = pred.argmax(1)
    pred = char[str(pred[0])]
    return pred

def detect(img, digit,num, idx):
    # if digit != '-1': return

    gray = img

    gray_mean=np.mean(gray)
    gray[gray< gray_mean*0.5] =0
    gray_mean=np.mean(gray)
    gray = cv2.add(gray_mean*0.1,gray)
    
    try:
        crop = np.where(gray>gray_mean,255,0).astype(np.uint8)
    except:
        print(f'{num}_{idx}')
        return    
    width = crop.shape[1]
    height = crop.shape[0]
    padding_width = (64 - width) // 2 
    padding_height = (64 - height) // 2 
    
    # Apply padding and resize
    crop = cv2.copyMakeBorder(crop, padding_height, padding_height, padding_width, padding_width, cv2.BORDER_CONSTANT, None, 255)
    crop = cv2.resize(crop, (64,64))
    cv2.imwrite(f"./total_data/{digit}/{num}_{idx}.png", crop)

def get_minus():
    return minus_data

def combination(pre_x,pre_y,pre_w,pre_h, x,y,w,h):
	if (abs(pre_x - x)<=10 and pre_y > y):   #(abs(kor[0][0] - kor[1][0])<=5 and kor[0][1] > kor[1][1]): 
		c_y=y
		y_h=pre_y+pre_h
		c_x=x
		x_w=pre_x + pre_w
	else:
		c_y= pre_y
		y_h= y+h
		c_x=pre_x
		x_w= x+w
	c_h= y_h -c_y
	c_w = x_w-c_x
	return c_x, c_y, c_w, c_h

def overlap(pre_x,pre_w, x,w):
    if (pre_x+pre_w - x)/(x+w - pre_x) >0.8:
        return True


######################################
	# #7개 이상 찾지 못한 경우 : 윤곽선을 찾는다.
	# if len(final_contour) <7:
	# 	gray = cv2.GaussianBlur(image, (1, 1), 0)
	# 	edged = cv2.Canny(gray, 50, 300)    
	# 	(contours, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	# 	contours = sorted([(c, cv2.boundingRect(c)[0]) for c in contours], key=lambda x: x[1])

	# 	####test
	# 	contour_y = list()
	# 	contour_w = list()
	# 	contour_h = list()
	# 	area =list()
	# 	final_contour=list()
	# 	previous_x_w =0
	# 	previous_y =0
	# 	for (contour, _) in contours:
	# 		# Compute the bounding box for the rectangle
	# 		(x, y, w, h) = cv2.boundingRect(contour)
	# 		if h >3 and h/w<5 and w/h<3 and x>=0 and x+w<=image_x:
	# 			if (x<12 or x>180) and (x+w>=image_x or abs(w-h)<=6 or h<15) and w+h<50: continue 
	# 			if (x<12  and w<16): continue #and abs(w-h)<=10
	# 			if (w>50): continue
	# 			if (x>180 and w<15 and h<30): continue
	# 			if (x<12 and y+h>=image_y): continue
	# 			if (w<10 or h <10): continue
	# 			if (x <= previous_x_w and y >=previous_y): continue
	# 			if (x <5 and h>50): continue
	# 			if (h <10): continue
	# 			if (x>210 and x+w>=image_x and y+h >=image_y): continue
	# 			if ((x< 10 or x >220) and (h <=10 or y+h == image_y)): continue
	# 			if (x <5 and h/w>6): continue
	# 			if (x <5 and h+w<51): continue
	# 			if (x <20 and h+w<51): continue

	# 			previous_x_w =x+w
	# 			previous_y =y
	# 			final_contour.append(contour)
	# 	w_mean = np.mean(contour_w)