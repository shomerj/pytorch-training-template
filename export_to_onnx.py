import cv2
import sys
import json
import torch
import argparse
import numpy as np
import onnxruntime as ort
from torchvision.ops._register_onnx_ops import _onnx_opset_version

sys.path.append('../')
from model.model import *

def main(args):
    cfg = json.load(open(args.cfg))
    cfg['model']['export'] = True
    model = build_model(cfg)
    checkpoint = torch.load(args.model_path)
    if 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
    model.load_state_dict(checkpoint)

    model_input_name = ['input']
    model_output_name = ['output']
    model_input = torch.randn(1,3,args.inp_size,args.inp_size)

    if args.name is None:
        name = 'Detector.onnx'
    else:
        name = f'{args.name}.onnx'

    torch.onnx.export(model.eval(), model_input, f'current_onnx_model/{name}',
            verbose=True,input_names=model_input_name,do_constant_folding=True,
            output_names=model_output_name, opset_version=_onnx_opset_version)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path')
    parser.add_argument('--cfg', help='path to config' )
    parser.add_argument('--inp_size', default=300, type=int)
    parser.add_argument('--name', default=None)
    args = parser.parse_args()
    main(args)
