#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for predicting intermediate outputs given a path to model and a csv file of domains
"""

from predict import predict_outputs
import argparse
import os
#from AlphaFold import AlphaFold
from Inception_aux import Inception_aux
#from ConvNet_aux import ConvNet_aux
import pickle
import torch
import numpy as np

parser = argparse.ArgumentParser('Predict intermediate outputs of a domain from AlphaFold and Inception Net')

parser.add_argument('-d', '--domain', metavar='', required=True, help='domain name')
parser.add_argument('-o', '--outputdir', metavar='', required=True, help='Directory where results should be saved')

args = parser.parse_args()

if __name__ == '__main__':
    
    # LOAD MODELS
    #alphafold = AlphaFold()
    inception = Inception_aux()
    #convnet = ConvNet_aux()
    
    #sd = torch.load('../../steps/alphafold_results/model.pth', map_location=torch.device('cpu'))
    #alphafold.load_state_dict(sd['model'])
    
    sd = torch.load('../../steps/inception_results/model.pth', map_location=torch.device('cpu'))
    inception.load_state_dict(sd['model'])
    
    #sd = torch.load('../../steps/convnet_results/model.pth', map_location=torch.device('cpu'))
    #convnet.load_state_dict(sd['model'])
    
    # PREDICT
    #p = predict_outputs(alphafold, args.domain)
    #with open(f'{args.outputdir}/alphafold/{args.domain}_out.pkl', 'wb') as f:
    #    pickle.dump(p, f)
    
    #p = predict_outputs(inception, args.domain)
    #with open(f'{args.outputdir}/inception/{args.domain}_out.pkl', 'wb') as f:
    #    pickle.dump(p, f)
    
    p = predict_outputs(inception, args.domain)
    with open(f'{args.outputdir}/{args.domain}_out.pkl', 'wb') as f:
        pickle.dump(p, f)
    
    #p = predict_outputs(convnet, args.domain)
    #with open(f'{args.outputdir}/convnet/{args.domain}_out.pkl', 'wb') as f:
    #    pickle.dump(p, f)
    