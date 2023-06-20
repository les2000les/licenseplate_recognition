# from flask import Flask, render_template, request, url_for, session, redirect
# from werkzeug.utils import secure_filename
# import os
# import requests
#
# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'static'  # 이미지 업로드 폴더 설정
# app.secret_key = 'secret_key'
#
# # 업로드 페이지 html 렌더링
# @app.route('/')
# def upload_file():
#     return render_template('file_upload.html')
#
# # 파일 업로드 처리
# @app.route('/uploader', methods=['GET', 'POST'])
# def uploader_file():
#     if request.method == 'POST':
#         f = request.files['file']
#         # 이미지 저장경로 지정
#         filename = secure_filename(f.filename)
#         upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         f.save(upload_path)
#         image_url = url_for('static', filename=f'file_upload/{filename}')  # 이미지 URL 생성
#         # POST 요청으로 이미지 데이터 전송
#         response = requests.post('http://127.0.0.1:5000/upload', files={'image': open(upload_path, 'rb')})
#
#         # 응답 확인
#         if response.status_code == 200:
#             print('이미지 전송 성공')
#             # return redirect(url_for('result'))
#             # return redirect(url_for('result_page'))
#             return redirect(url_for('display_result'))
#         else:
#             print(response.status_code)
#             print('이미지 전송 실패')
#             # return render_template('result.html', image_url=image_url)
#             return "Fail"
#
# # # 결과 이미지 표시
# @app.route('/result', methods=['POST', 'GET'])
# def display_result():
#     if request.method == 'POST':
#         # 이미지 파일 수신 대기
#         while 'image' not in request.files:
#             pass
#
#         image_file = request.files['image']
#         # 결과 이미지 저장
#         save_path = os.path.join(app.config['UPLOAD_FOLDER'], 'getresult.jpg')
#         image_file.save(save_path)
#         print('결과 이미지 저장 완료')
#
#         # 이미지 URL 생성
#         image_url = url_for('static', filename='getresult.jpg')
#
#         # 세션에 이미지 URL 저장
#         session['image_url'] = image_url
#
#         return redirect(url_for('result_page'))
#     else:
#         # 이미지 파일이 전송되지 않은 경우에는 실패로 간주합니다.
#         print('결과 이미지 저장 실패')
#         return 'Image not found in request.'
#
# # 결과 페이지 html 렌더링
# @app.route('/result_page', methods=['POST', 'GET'])
# # def result_page():
# #     save_path = 'static/getresult.jpg'
# #
# #     # 전송된 이미지 데이터 받기
# #     image_file = request.files['image']
# #
# #     # 이미지 파일 저장
# #     image_file.save(save_path)
# #     print(image_file)
# #     print('결과 이미지 저장완료')
# #
# #     # image_url = url_for('static', filename='getresult.jpg')
# #     return render_template('result.html', result_image_url=save_path)
#
# def result_page():
#     if 'image_url' in session:
#         image_url = session['image_url']
#         return render_template('result.html', image_url=image_url)
#     else:
#         return 'Image URL not found in session.'
#
# if __name__ == '__main__':
#     app.run(debug=True, port=9999)


from flask import Flask, render_template, request, url_for, session, redirect
from werkzeug.utils import secure_filename
import os
import requests
import threading

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'  # 이미지 업로드 폴더 설정
app.secret_key = 'secret_key'
app.config['SERVER_NAME'] = 'localhost:9999'  # 서버 이름 설정
app.config['APPLICATION_ROOT'] = '/'  # 응용 프로그램 루트 설정

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
        # POST 요청으로 이미지 데이터 전송
        response = requests.post('http://127.0.0.1:5000/upload', files={'image': open(upload_path, 'rb')})

        # 응답 확인
        if response.status_code == 200:
            print('이미지 전송 성공')
            # 이미지 전송 성공한 경우에만 결과 이미지 저장
            threading.Thread(target=save_result_image, args=(upload_path,)).start()
            return redirect(url_for('display_result'))
        else:
            print(response.status_code)
            print('이미지 전송 실패')
            return "Fail"

# 결과 이미지 저장
def save_result_image(upload_path):
    with app.app_context():
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], 'getresult.jpg')
        # 이미지 파일을 직접 열어 읽어서 저장합니다.
        with open(upload_path, 'rb') as image_file:
            with open(save_path, 'wb') as result_file:
                result_file.write(image_file.read())
        print('결과 이미지 저장 완료')

        # 이미지 URL 생성
        with app.test_request_context():
            image_url = url_for('static', filename='getresult.jpg')

        # 세션에 이미지 URL 저장
        session['image_url'] = image_url

# # 결과 이미지 표시
@app.route('/result', methods=['POST', 'GET'])
def display_result():
    if request.method == 'POST':
        if 'image_url' in session:
            image_url = session['image_url']
            return render_template('result.html', image_url=image_url)
        else:
            return 'Image URL not found in session.'
    else:
        # 이미지 파일이 전송되지 않은 경우에는 실패로 간주합니다.
        print('결과 이미지 저장 실패')
        return 'Image not found in request.'

if __name__ == '__main__':
    app.run(debug=True, port=9999)