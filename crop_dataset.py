import os
import argparse
import glob
import numpy as np
from PIL import Image
from torchvision.transforms.functional import center_crop
import torch
from tqdm import tqdm
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='모든 이미지를 크롭하는 스크립트')
    parser.add_argument('--data_path', type=str, default='data/shaman_1', help='데이터 경로')
    parser.add_argument('--output_path', type=str, default='data/shaman_1_cropped', help='크롭된 이미지 저장 경로')
    parser.add_argument('--crop_size', type=str, default='432_864', help='크롭 크기 (H_W)')
    parser.add_argument('--resize', type=str, default='-1', help='리사이즈 크기 (H_W 또는 단일 숫자)')
    parser.add_argument('--center_crop', action='store_true', help='중앙 크롭 사용')
    parser.add_argument('--overwrite', action='store_true', help='출력 디렉토리가 존재하면 덮어쓰기')
    return parser.parse_args()

def create_directory(path):
    """디렉토리가 없으면 생성"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"디렉토리 생성: {path}")

def process_image(img_path, output_path, crop_size, resize, center_crop_flag=True):
    """이미지를 로드, 크롭, 저장"""
    # 이미지 로드
    img = Image.open(img_path)
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    
    # 크롭 크기 파싱
    if crop_size != '-1':
        crop_h, crop_w = [int(x) for x in crop_size.split('_')[:2]]
        if center_crop_flag:
            # 중앙 크롭
            img_tensor = center_crop(img_tensor, (crop_h, crop_w))
        else:
            # 좌상단 크롭 (필요시 구현)
            h, w = img_tensor.shape[1:]
            img_tensor = img_tensor[:, :crop_h, :crop_w]
    
    # 리사이즈
    if resize != '-1':
        if '_' in resize:
            resize_h, resize_w = [int(x) for x in resize.split('_')]
            img_tensor = torch.nn.functional.interpolate(
                img_tensor.unsqueeze(0), 
                size=(resize_h, resize_w), 
                mode='bicubic', 
                align_corners=False
            ).squeeze(0)
        else:
            resize_hw = int(resize)
            img_tensor = torch.nn.functional.interpolate(
                img_tensor.unsqueeze(0), 
                size=(resize_hw, resize_hw), 
                mode='bicubic', 
                align_corners=False
            ).squeeze(0)
    
    # 텐서를 PIL 이미지로 변환
    img_np = (img_tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    
    # 저장
    img_pil.save(output_path)

def main():
    args = parse_args()
    
    # 출력 디렉토리 처리
    if os.path.exists(args.output_path):
        if args.overwrite:
            print(f"기존 디렉토리를 덮어씁니다: {args.output_path}")
            shutil.rmtree(args.output_path)
        else:
            print(f"출력 디렉토리가 이미 존재합니다: {args.output_path}")
            print("덮어쓰려면 --overwrite 옵션을 사용하세요.")
            return
            
    # 처리할 이미지 리스트 가져오기
    subfolders = sorted(os.listdir(args.data_path))
    all_images = []
    
    # 모든 폴더 구조 및 이미지 경로 찾기
    for folder in subfolders:
        folder_path = os.path.join(args.data_path, folder)
        if os.path.isdir(folder_path):
            # 출력 폴더 생성
            output_folder = os.path.join(args.output_path, folder)
            create_directory(output_folder)
            
            # 폴더 내 모든 이미지 찾기
            images = sorted(glob.glob(os.path.join(folder_path, '*.png')))
            for img_path in images:
                img_name = os.path.basename(img_path)
                output_path = os.path.join(output_folder, img_name)
                all_images.append((img_path, output_path))
    
    print(f"총 {len(all_images)}개의 이미지를 처리합니다.")
    
    # 각 이미지 처리
    for img_path, output_path in tqdm(all_images):
        process_image(
            img_path, 
            output_path, 
            args.crop_size, 
            args.resize, 
            args.center_crop
        )
    
    print(f"완료! 크롭된 이미지가 {args.output_path}에 저장되었습니다.")

if __name__ == "__main__":
    main() 