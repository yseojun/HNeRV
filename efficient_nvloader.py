import torch
import os
import shutil
from tqdm import tqdm
import argparse
import time
import pandas as pd
import numpy as np
from torchvision.utils import save_image
from torchvision.io import write_video
from model_all import PositionEncoding

def dequant_tensor(quant_t):
    quant_t, tmin, scale = quant_t['quant'], quant_t['min'].to(torch.float32), quant_t['scale'].to(torch.float32)
    new_t = tmin.expand_as(quant_t) + scale.expand_as(quant_t) * quant_t
    return new_t
    
class VideoRenderer:
    def __init__(self, ckt, decoder, pe=True, grid_size=9, device='cuda:0', lbase=2.0, levels=32):
        self.grid_size = grid_size
        self.pe = pe
        
        if pe:
            self.pe_embed = PositionEncoding(f'pe_{lbase}_{levels}')
        
        quant_ckt = torch.load(ckt, map_location='cpu')
        self.vid_embed = dequant_tensor(quant_ckt['embed']).to(device)

        self.frames = self.vid_embed.size(0) // (grid_size * grid_size)

        dequant_ckt = {k:dequant_tensor(v).to(device) for k,v in quant_ckt['model'].items()}

        self.decoder = torch.jit.load(decoder, map_location='cpu').to(device)
        self.decoder.load_state_dict(dequant_ckt)

    def calculate_index(self, y, x, frame):
        pos = y * self.grid_size + x
        return pos * self.frames + frame

    def get_interpolated_vid_embed(self, start_x, start_y, frame):
        x_remain = start_x % 1
        y_remain = start_y % 1
        frame_remain = frame % 1

        x_int = int(start_x)
        y_int = int(start_y)
        frame_int = int(frame)

        idx_000 = self.calculate_index(y_int, x_int, frame_int)
        idx_100 = self.calculate_index(y_int, x_int+1, frame_int)
        idx_010 = self.calculate_index(y_int+1, x_int, frame_int)
        idx_110 = self.calculate_index(y_int+1, x_int+1, frame_int)
        idx_001 = self.calculate_index(y_int, x_int, frame_int+1)
        idx_101 = self.calculate_index(y_int, x_int+1, frame_int+1)
        idx_011 = self.calculate_index(y_int+1, x_int, frame_int+1)
        idx_111 = self.calculate_index(y_int+1, x_int+1, frame_int+1)
        
        w_000 = (1-x_remain) * (1-y_remain) * (1-frame_remain)
        w_100 = x_remain * (1-y_remain) * (1-frame_remain)
        w_010 = (1-x_remain) * y_remain * (1-frame_remain)
        w_110 = x_remain * y_remain * (1-frame_remain)
        w_001 = (1-x_remain) * (1-y_remain) * frame_remain
        w_101 = x_remain * (1-y_remain) * frame_remain 
        w_011 = (1-x_remain) * y_remain * frame_remain
        w_111 = x_remain * y_remain * frame_remain

        result = w_000 * self.vid_embed[idx_000] + \
                 w_100 * self.vid_embed[idx_100] + \
                 w_010 * self.vid_embed[idx_010] + \
                 w_110 * self.vid_embed[idx_110] + \
                 w_001 * self.vid_embed[idx_001] + \
                 w_101 * self.vid_embed[idx_101] + \
                 w_011 * self.vid_embed[idx_011] + \
                 w_111 * self.vid_embed[idx_111]

        return result

    def get_vid_embed(self, start_x, start_y, end_x, end_y, frames):
        x_step = float(end_x - start_x) / frames
        y_step = float(end_y - start_y) / frames
        frame_step = self.frames / frames
        vid_embed_list = []

        for i in range(frames):
            f = i * frame_step
            print(f"프레임 {i}: x={start_x:.2f}, y={start_y:.2f}, f={f:.2f}")
            if start_x % 1 != 0 or start_y % 1 != 0 or f % 1 != 0:
                vid_embed_list.append(self.get_interpolated_vid_embed(start_x, start_y, f))
            else:
                vid_embed_list.append(self.vid_embed[self.calculate_index(int(start_y), int(start_x), int(f))])
            start_x += x_step
            start_y += y_step
            
        return torch.stack(vid_embed_list)

    def get_norm_input(self, start_x, start_y, end_x, end_y, frames):
        decoder_device = next(self.decoder.parameters()).device
        
        norm_y = (torch.linspace(start_y, end_y, frames, device=decoder_device) / self.grid_size).unsqueeze(1)
        norm_x = (torch.linspace(start_x, end_x, frames, device=decoder_device) / self.grid_size).unsqueeze(1)
        norm_idx = torch.linspace(0, 1, frames, device=decoder_device).unsqueeze(1)
        
        for i in range(frames):
            y_val = norm_y[i].item()
            x_val = norm_x[i].item()
            idx_val = norm_idx[i].item()
            print(f"프레임 {i}: y={y_val:.2f}, x={x_val:.2f}, idx={idx_val:.2f}")
        return torch.cat([norm_y, norm_x, norm_idx], dim=1)

    def render(self, start_x, start_y, end_x, end_y, frames):
        if not self.pe:
            input = self.get_vid_embed(start_x, start_y, end_x, end_y, frames)
        else:
            pe_input = self.get_norm_input(start_x, start_y, end_x, end_y, frames)
            input = self.pe_embed(pe_input)
            
        output = self.decoder(input)
        return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--decoder', type=str, default='checkpoints/img_decoder.pth', help='path for video decoder',)
    parser.add_argument('--ckt', type=str, default='checkpoints/quant_vid.pth', help='path for video checkpoint',)
    parser.add_argument('--dump_dir', type=str, default='visualize/bunny_1.5M_E300', help='path for video checkpoint',)
    parser.add_argument('--frames', type=int, default=16, help='video frames for output',)
    parser.add_argument('--grid_size', type=int, default=9, help='grid size',)
    parser.add_argument('--pe', type=bool, default=False, help='use pe',)
    parser.add_argument('--lbase', type=float, default=2.0, help='base for position encoding',)
    parser.add_argument('--levels', type=int, default=32, help='levels for position encoding',)
    args = parser.parse_args()
    
    if not os.path.exists(args.dump_dir):
        os.makedirs(args.dump_dir)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    video_renderer = VideoRenderer(args.ckt, args.decoder, args.pe, args.grid_size, device, args.lbase, args.levels)

    if device.startswith('cuda'):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        print("CUDA 타이머 시작")

    if device.startswith('cuda'):
        start_event.record()
        
    img_out = video_renderer.render(0, 0, 0, 0, args.frames).cpu()
    
    if device.startswith('cuda'):
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        print(f"GPU 연산 시간: {elapsed_time:.2f} 밀리초")

    out_vid = os.path.join(args.dump_dir, 'nvloader_out.mp4')
    write_video(out_vid, img_out.permute(0,2,3,1) * 255., args.frames/4, options={'crf':'10'})

    print(f'dumped video to {out_vid}')

if __name__ == '__main__':
    main()
