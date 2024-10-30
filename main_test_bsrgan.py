import os.path
import logging
import torch
from utils import utils_logger
from utils import utils_image as util
from models.network_rrdbnet import RRDBNet as net

def main():
    utils_logger.logger_info('blind_sr_log', log_path='blind_sr_log.log')
    logger = logging.getLogger('blind_sr_log')

    testsets = 'testsets'       # 테스트셋 경로 설정
    testset_Ls = ['RealSRSet']  # 사용할 테스트셋 목록
    model_names = ['BSRGAN']    # 모델 목록 (여기서는 BSRGAN만 사용)

    save_results = True
    sf = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_name in model_names:
        if model_name in ['BSRGANx2']:
            sf = 2
        model_path = os.path.join('model_zoo', model_name+'.pth')
        logger.info('{:>16s} : {:s}'.format('Model Name', model_name))

        # GPU 설정 메시지
        if device.type == 'cuda':
            logger.info('{:>16s} : GPU'.format('Device'))
        else:
            logger.info('{:>16s} : CPU only'.format('Device'))

        # 모델 정의 및 로드
        model = net(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=sf)
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        model.eval()
        for k, v in model.named_parameters():
            v.requires_grad = False
        model = model.to(device)

        for testset_L in testset_Ls:
            L_path = os.path.join(testsets, testset_L)
            E_path = os.path.join(testsets, f"{testset_L}_results_x{sf}")
            util.mkdir(E_path)

            logger.info('{:>16s} : {:s}'.format('Input Path', L_path))
            logger.info('{:>16s} : {:s}'.format('Output Path', E_path))
            idx = 0

            for img in util.get_image_paths(L_path):
                idx += 1
                img_name, ext = os.path.splitext(os.path.basename(img))
                logger.info('{:->4d} --> {:<s} --> x{:<d} --> {:<s}'.format(idx, model_name, sf, img_name+ext))

                # 이미지 로드 및 처리
                img_L = util.imread_uint(img, n_channels=3)
                img_L = util.uint2tensor4(img_L).to(device)

                # 추론 수행
                with torch.no_grad():
                    img_E = model(img_L)

                # 결과 저장
                img_E = util.tensor2uint(img_E)
                if save_results:
                    util.imsave(img_E, os.path.join(E_path, img_name+'_'+model_name+'.png'))

if __name__ == '__main__':
    main()
