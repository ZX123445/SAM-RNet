import os

import cv2
from tqdm import tqdm
import sod_metrics as M

FM = M.Fmeasure()
WFM = M.WeightedFmeasure()
SM = M.Smeasure()
EM = M.Emeasure()
MAE = M.MAE()


for _data_name in ['CAMO', 'COD10K', 'NC4K', 'CHAMELEON']:
    mask_root = '/hy-tmp/C/TestDataset/{}/GT'.format(_data_name)
    pred_root = '/hy-tmp/WS/res_10_384/snap_10_384_迭代/{}/'.format(_data_name)
    mask_name_list = sorted(os.listdir(mask_root))

    for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
        mask_path = os.path.join(mask_root, mask_name)
        pred_path = os.path.join(pred_root, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        FM.step(pred=pred, gt=mask)
        WFM.step(pred=pred, gt=mask)
        SM.step(pred=pred, gt=mask)
        EM.step(pred=pred, gt=mask)
        MAE.step(pred=pred, gt=mask)

    fm = FM.get_results()['fm']
    wfm = WFM.get_results()['wfm']
    sm = SM.get_results()['sm']
    em = EM.get_results()['em']
    mae = MAE.get_results()['mae']



    print(
        'Smeasure:', sm.round(3), '; ',
        'wFmeasure:', wfm.round(3), '; ',
        'MAE:', mae.round(3), '; ',
        'adpEm:', em['adp'].round(3), '; ',
        'meanEm:', '-' if em['curve'] is None else em['curve'].mean().round(3), '; ',
        'maxEm:', '-' if em['curve'] is None else em['curve'].max().round(3), '; ',
        'adpFm:', fm['adp'].round(3), '; ',
        'meanFm:', fm['curve'].mean().round(3), '; ',
        'maxFm:', fm['curve'].max().round(3),
        sep=''
    )



    with open("../result_10-384_分辨率迭代机制.txt", "a+") as f:
        print('Smeasure:', sm.round(3), '; ',
               'meanEm:', '-' if em['curve'] is None else em['curve'].mean().round(3), '; ',
              'maxEm:', '-' if em['curve'] is None else em['curve'].max().round(3), '; ',
              'wFmeasure:', wfm.round(3), '; ',
              'MAE:', mae.round(3), '; ',
              'meanFm:', fm['curve'].mean().round(3), '; ',
              'maxFm:', fm['curve'].max().round(3),
              file=f
              )
