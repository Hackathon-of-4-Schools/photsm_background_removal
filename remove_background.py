import torch
import torch.nn as nn
import cv2
import os
import numpy as np
from torchvision import models, transforms
from PIL import Image

# 배경 교체 함수
def ReplaceBG(image, mask):
    # 이미지와 마스크를 NumPy 배열로 변환
    image_np = np.array(image)

    # 마스크를 이진화하여 배경을 교체
    mask_np = (mask > 0).astype(np.uint8)  # 마스크의 픽셀값을 0과 1로 변환
    mask_np = np.repeat(mask_np[:, :, np.newaxis], 3, axis=2)  # 3채널로 확장

    # 투명 배경 생성
    result = np.zeros_like(image_np, dtype=np.uint8)
    result[..., :3] = image_np * mask_np  # 이미지 픽셀 설정 (마스크 영역)
    
    # 알파 채널 추가
    alpha_channel = np.where(mask_np[..., 0] > 0, 255, 0).astype(np.uint8)
    result = np.dstack((result, alpha_channel))  # RGBA 이미지로 변환

    return Image.fromarray(result, 'RGBA')

# BodyDetector 클래스 구현 (탐지 모델)
class BodyDetector:
    def __init__(self):
        # Mask R-CNN 모델 로드
        self.model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()  # 평가 모드로 설정

    def DetectBody(self, img):
        # 이미지 변환
        transform = transforms.Compose([
            transforms.ToTensor(),  # Tensor로 변환
        ])
        img_tensor = transform(img).unsqueeze(0)  # 배치 차원 추가

        with torch.no_grad():
            predictions = self.model(img_tensor)  # 모델 예측

        # 첫 번째 예측에서 사람 클래스(클래스 ID 1)에 대한 마스크 선택
        masks = predictions[0]['masks']
        labels = predictions[0]['labels']
        mask = masks[labels == 1]  # 사람 클래스에 해당하는 마스크만 선택

        if mask.size(0) > 0:  # 사람이 감지되면
            return (mask[0, 0].cpu().numpy() > 0.5).astype(np.uint8)  # 첫 번째 마스크 반환 (이진화)
        else:
            return np.zeros((img.size[1], img.size[0]), dtype=np.uint8)  # 감지된 사람이 없으면 빈 마스크 반환

# 이미지 로드 함수
def LoadImage(filename):
    image = cv2.imread(filename)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR에서 RGB로 변환

# 메인 코드 실행
if __name__ == "__main__":
    # 모델 로드
    detector = BodyDetector()

    # 사용자로부터 입력 이미지 경로를 받음
    img_fname = "input.png"
    
    # 입력 이미지 읽기
    img = LoadImage(img_fname)
    img_pil = Image.fromarray(img)

    # 객체 탐지
    mask = detector.DetectBody(img_pil)

    # 배경 교체 (투명 배경)
    res = ReplaceBG(img_pil, mask)

    # 결과 이미지 저장
    output_path = "output.png"
    res.save(output_path)
    print(f"결과 이미지가 '{output_path}'로 저장되었습니다.")