import requests

# 이미지 파일 경로
image_path = 'file_upload/car.png' #예시 이미지 파일

# 파일 열기
with open(image_path, 'rb') as file:
    # 파일 데이터 읽기
    file_data = file.read()

# POST 요청으로 이미지 데이터 전송
response = requests.post('http://127.0.0.1:5000/upload', files={'image': file_data})

# 응답 확인
if response.status_code == 200:
    print('이미지 전송 성공')
else:
    print('이미지 전송 실패')