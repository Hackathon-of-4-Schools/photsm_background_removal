import os.path
import logging
import torch
from utils import utils_logger
from utils import utils_image as util
from models.network_rrdbnet import RRDBNet as net

def enhance_image(input_image_path, model_name='BSRGAN', sf=4):
    # 로거 설정
    utils_logger.logger_info('blind_sr_log', log_path='blind_sr_log.log')
    logger = logging.getLogger('blind_sr_log')

    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 모델 경로와 로드
    model_path = os.path.join('model_zoo', model_name + '.pth')
    model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=sf)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    model.eval()
    model = model.to(device)

    # 입력 이미지 로드 및 전처리
    img_L = util.imread_uint(input_image_path, n_channels=3)
    img_L = util.uint2tensor4(img_L).to(device)

    # 화질 개선 수행
    with torch.no_grad():
        img_E = model(img_L)

    # 결과 이미지 후처리 및 반환
    img_E = util.tensor2uint(img_E)
    return img_E  # 개선된 이미지 반환

# 사용 예시
input_path = 'image.png'
output_image = enhance_image(input_path)
util.imsave(output_image, 'enhanced_image.png')
