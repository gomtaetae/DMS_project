import os
import cv2
import time
import dlib
import pygame
import psutil
import imutils
import datetime
import collections
from imutils import face_utils
from scipy.spatial import distance
from facial_tracking.eye import Eye
import numpy as np
import facial_tracking.conf as conf
from facial_tracking.iris import Iris
from facial_tracking.faceMesh import FaceMesh
# from server.pybo.dataSave import save_file_info_to_db


class FacialProcessor:
	def __init__(self):
		self.face_mesh = FaceMesh()
		self.detected = False
		self.eyes_status = "Eyes not detected"
		self.face_mesh_enabled = True  # FaceMesh 처리를 기본적으로 활성화

		# face mesh toggle 키 만들기
	def toggle_face_mesh(self):
		self.face_mesh_enabled = not self.face_mesh_enabled

	def process_frame(self, frame):
		if self.face_mesh_enabled:
		# FaceMesh 처리 코드
			self.face_mesh.process_frame(frame)

			# 원본 프레임에 대한 다른 처리 (여기에 코드 추가)
			self.face_mesh.frame = frame
			self.face_mesh.draw_mesh()

			# Add your logic to detect eyes and update self.detected and self.eyes_status
			# For demonstration purposes, let's assume eyes are always detected.
			self.detected = True
			self.eyes_status = "Eyes detected"

			if self.face_mesh.mesh_result.multi_face_landmarks:
				for face_landmarks in self.face_mesh.mesh_result.multi_face_landmarks:
					left_iris = Iris(frame, face_landmarks, conf.LEFT_EYE)
					right_iris = Iris(frame, face_landmarks, conf.RIGHT_EYE)
					left_iris.draw_iris(True)
					right_iris.draw_iris(True)

					# Add eye-tracking logic here
					left_eye = Eye(frame, face_landmarks, conf.LEFT_EYE)
					right_eye = Eye(frame, face_landmarks, conf.RIGHT_EYE)
					left_eye.iris.draw_iris()
					right_eye.iris.draw_iris()

					if left_eye.eye_closed() or right_eye.eye_closed():
						self.eyes_status = 'Eye closed'
					else:
						if left_eye.gaze_right() and right_eye.gaze_right():
							self.eyes_status = 'Gazing right'
						elif left_eye.gaze_left() and right_eye.gaze_left():
							self.eyes_status = 'Gazing left'
						elif left_eye.gaze_center() and right_eye.gaze_center():
							self.eyes_status = 'Gazing center'

			# Draw the face mesh after drawing the iris
			self.face_mesh.draw_mesh()


def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

# 이벤트 발생시 캡처화면 생성
def save_frame(frame):
	# flipped_frame = cv2.flip(frame, 1)  # 좌우 반전
	# 현재 시각을 기반으로 파일 이름을 생성한다.
	timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
	filename = f"capture_{timestamp}.jpg"
	filepath = os.path.join("save/capture", filename) # 저장할 경로 지정
	cv2.imwrite(filepath, frame)
	return filepath


# 이벤트 발생 시 영상 저장을 시작
def start_saving_event_video(frame_width, frame_height):
    # 현재 시각을 기반으로 파일 이름을 생성
	global event_occurred, event_frame_count
	event_occurred = True
	event_frame_count = 0
	timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
	filename = f"event_{timestamp}.mp4"
	# 저장할 경로 설정
	save_path = os.path.join("save/videos", filename)
	video_writer = cv2.VideoWriter(save_path, fourcc, fps, (frame_width, frame_height))
	while frames_buffer:
		buffer_frame = cv2.flip(frames_buffer.popleft(), 1)  # 버퍼 프레임 반전
		video_writer.write(buffer_frame)
	return video_writer

thresh = 0.2
frame_check = 15
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

detect = dlib.get_frontal_face_detector()
path = 'C:/Users/KOSA/Downloads/DMS_project/ai-model/'
predict = dlib.shape_predictor(path + './haar/shape_predictor_68_face_landmarks.dat')
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("C:/Users/KOSA/Downloads/DMS_project/ai-model/sound/alarm.wav")

flag = 0
facial_processor = FacialProcessor()
ptime = 0
ctime = 0
leftEAR = 0.0
rightEAR = 0.0


# 동영상 저장을 위한 프레임 설정
buffer_size = 2 # 이벤트 시작전 2초
fps = 30 # 초당 프레임 수
fourcc = cv2.VideoWriter_fourcc(*'MP4V') # 비디오 코덱 설정

# 프레임 버퍼 초기화
frames_buffer = collections.deque(maxlen=buffer_size * fps) # 프레임을 저장할 수

# 이벤트 발생 시 영상을 저장할 준비
event_occurred = False
event_frame_count = 0 # 이벤트 발생 후 프레임 수

# 이벤트 후 녹화할 추가 시간 설정(5초)
extra_recording_time = 5 * fps  # 5초( * fps는 프레임 단위로 변경)
post_event_frames = buffer_size * fps # 이벤트 후 저장할 프레임 수


# Adjust the capture device index (0 or 1) based on your camera setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Set the width of the frames
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1240)  # Set the height of the frames
cap.set(cv2.CAP_PROP_FPS, 30)  # Set the frames per second


# 졸음 감지 후 지속적인 알람을 위한 추가 변수 설정
continuous_alarm_count = 0
alarm_interval = 30  # 알람 간격 (프레임 수 기준)

# 캡처 및 영상 녹화 ON/OFF
capture_enabled = True
recording_enabled = True

# 캡처 및 영상 녹화 상태 출력
capture_status_text = "Capture ON"
recording_status_text = "Record ON"


while True:
	ret, frame = cap.read()
	# ret이 false일 경우(카메라에 문제가 생겨 꺼졌거나 프레임 반응이 없을때) 무한루프에 돌지 않도록 종료(break)시켜주는 코드
	if not ret:
		break

	# CPU 사용률 가져오기
	cpu_usage = psutil.cpu_percent()

	flipped_frame = cv2.flip(frame, 1)

	frames_buffer.append(flipped_frame)  # 프레임 버퍼에 뒤집힌 프레임 추가


	# Check if frame is not None before processing
	if frame is not None:
		frame = imutils.resize(flipped_frame, width=1020)
		gray = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2GRAY)

		# Use the face detector
		subjects = detect(gray)

		for subject in subjects:
			shape = predict(gray, subject)
			shape = face_utils.shape_to_np(shape)

			leftEye = shape[lStart:lEnd]
			rightEye = shape[rStart:rEnd]
			leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye)
			ear = (leftEAR + rightEAR) / 2.0

			if ear < thresh:
				flag += 1
				print(flag)

				if flag == frame_check:
					cv2.putText(frame, "", (10, 30),
					            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

					# 캡처 진행
					if capture_enabled:
						capture_status_text = "Capture ON" if capture_enabled else "Capture OFF"
						# 캡처하고 파일 경로를 반환받는다.
						capture_path = save_frame(frame)
						print(f"Capture image saved to: {capture_path}")
					else:
						capture_status_text = "Capture OFF"

					# 경고 알람 재생
					alarm_sound.play()
					print("Drowsiness Detected")

					# 영상 이벤트 진행
					if recording_enabled:
						recording_status_text = "Record ON" if recording_enabled else "Record OFF"
						# 이벤트 발생 시점이면, 동영상 저장 준비
						if not event_occurred:
							frame_width = frame.shape[1]
							frame_height = frame.shape[0]
							video_path = start_saving_event_video(frame_width, frame_height)
							# 버퍼에 저장된 프레임(이벤트 발생 전 2초)을 영상 파일에 먼저 기록
							while frames_buffer:
								video_path.write(frames_buffer.popleft())
							event_occurred = True
							event_frame_count = 0  # 프레임 카운트 초기화
					else:
						recording_status_text = "Record OFF"

				# 이벤트가 지속되는 동안 프레임을 기록
				if event_occurred and event_frame_count < post_event_frames:
					video_path.write(flipped_frame)
					event_frame_count += 1

				# 이벤트 후 추가 시간(5초)까지 녹화 완료
				elif event_occurred and event_frame_count >= post_event_frames:
					video_path.release()
					event_occurred = False
					frames_buffer.clear()

				# 사용자가 눈을 감고 있어도 지속적으로 알람을 주기
				if flag > frame_check:
					# 졸음 상태에 있을때 카운트를 1씩 증가
					continuous_alarm_count += 1

					# 졸음 카운트가 설정된 알람 간격에 도달했는지 체크 위에 30으로 설정
					if continuous_alarm_count >= alarm_interval:
						alarm_sound.play()
						print("Continuous Drowsiness Detected")
						# 알람을 울리면 카운트를 다시 0으로 초기화
						continuous_alarm_count = 0

			else:
				if event_occurred:
					if event_frame_count < (buffer_size * fps + extra_recording_time):
						# 이벤트 후 추가 시간 동안 계속 녹화
						video_path.write(frame)
						event_frame_count += 1
					else:
						# 추가 시간이 끝나면 녹화 종료
						video_path.release()
						event_occurred = False
						frames_buffer.clear()
				flag = 0
				if event_occurred and event_frame_count >= post_event_frames:
					video_path.release()
					event_occurred = False
					frames_buffer.clear()

					# 데이터베이스에 파일 정보 저장
					# user_id = 1  # 예시 사용자 ID, 실제 사용자 ID로 교체 필요
					# save_file_info_to_db(user_id, capture_path, video_path)

				# 버퍼를 다시 초기화하여 다음 이벤트 대비
				frames_buffer.clear()

		# FaceMesh 프로세싱 및 원본 프레임에 그리기
		facial_processor.process_frame(frame)

		# 새로운 프레임 생성 및 검은색으로 초기화
		face_mesh_frame = np.zeros_like(frame)

		# 원본 프레임에 대한 처리
		facial_processor.face_mesh.frame = frame
		facial_processor.face_mesh.draw_mesh()

		# FaceMesh Frame에 추가 정보 표시
		facial_processor.face_mesh.frame = face_mesh_frame
		facial_processor.face_mesh.draw_mesh()

		ctime = time.time()
		fps = 1 / (ctime - ptime)
		ptime = ctime


		# FaceMesh ON/OFF 설정
		frame_width = frame.shape[1]
		text = "FaceMesh ON" if facial_processor.face_mesh_enabled else "FaceMesh OFF"
		text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
		text_x = frame_width - text_size[0] - 10  # 오른쪽 여백 10 픽셀 고려
		text_y = 30  # 상단 여백

		cv2.putText(face_mesh_frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


		cv2.putText(face_mesh_frame, f'FPS: {int(fps)}', (30, 30), 0, 0.6,
					conf.TEXT_COLOR, 1, lineType=cv2.LINE_AA)

		cv2.putText(face_mesh_frame, f'{facial_processor.eyes_status}', (30, 70), 0, 0.8,
					conf.TEXT_COLOR, 2, lineType=cv2.LINE_AA)

		# Flip text rendering for the left and right ears
		cv2.putText(face_mesh_frame, "Left EAR {:.2f}".format(leftEAR), (30, 550),
		            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
		cv2.putText(face_mesh_frame, "Right EAR {:.2f}".format(rightEAR), (30, 570),
		            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

		# 화면에 CPU 사용률을 표시하기
		cv2.putText(face_mesh_frame, f'CPU Usage: {cpu_usage}%', (text_x, text_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

		# 화면에 텍스트 표시
		cv2.putText(face_mesh_frame, capture_status_text, (text_x, text_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
		cv2.putText(face_mesh_frame, recording_status_text, (text_x, text_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

		# 전체 통합 출력
		cv2.imshow('Original Frame', frame)
		cv2.imshow('FaceMesh Frame', face_mesh_frame)


	# Delay to control the frames per second
	key = cv2.waitKey(1) & 0xFF
	if key == ord('m') or key == ord('M'):
		facial_processor.toggle_face_mesh()

	if key == ord('c'):
		capture_enabled = not capture_enabled
		capture_status_text = "Capture ON" if capture_enabled else "Capture OFF"

	if key == ord('r'):
		recording_enabled = not recording_enabled
		recording_status_text = "Record ON" if recording_enabled else "Record OFF"

	if key == ord('q'):
		break

# Release the capture and close all windows
cv2.destroyAllWindows()
cap.release()