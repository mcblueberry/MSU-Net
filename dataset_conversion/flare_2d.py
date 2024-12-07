# import numpy as np
# import SimpleITK as sitk
# from utils import ResampleXYZAxis, ResampleLabelToRef, CropForeground
# import os
# import random
# import yaml
# import copy
# import pdb
#
# def ResampleImage(imImage, imLabel, save_path, name, target_spacing=(1., 1., 1.)):
#
#     assert round(imImage.GetSpacing()[0], 2) == round(imLabel.GetSpacing()[0], 2)
#     assert round(imImage.GetSpacing()[1], 2) == round(imLabel.GetSpacing()[1], 2)
#     assert round(imImage.GetSpacing()[2], 2) == round(imLabel.GetSpacing()[2], 2)
#
#     assert imImage.GetSize() == imLabel.GetSize()
#
#
#     spacing = imImage.GetSpacing()
#     origin = imImage.GetOrigin()
#
#     imLabel.CopyInformation(imImage)
#
#     npimg = sitk.GetArrayFromImage(imImage).astype(np.int32)
#     nplab = sitk.GetArrayFromImage(imLabel).astype(np.uint8)
#     z, y, x = npimg.shape
#
#     if not os.path.exists('%s'%(save_path)):
#         os.mkdir('%s'%(save_path))
#
#     imImage.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
#     imLabel.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
#
#
#     re_img_xy = ResampleXYZAxis(imImage, space=(target_spacing[0], target_spacing[1], spacing[2]), interp=sitk.sitkBSpline)
#     re_lab_xy = ResampleLabelToRef(imLabel, re_img_xy, interp=sitk.sitkNearestNeighbor)
#
#     re_img_xyz = ResampleXYZAxis(re_img_xy, space=(target_spacing[0], target_spacing[1], target_spacing[2]), interp=sitk.sitkNearestNeighbor)
#     re_lab_xyz = ResampleLabelToRef(re_lab_xy, re_img_xyz, interp=sitk.sitkNearestNeighbor)
#
#
#
#
#     cropped_img, cropped_lab = CropForeground(re_img_xyz, re_lab_xyz, context_size=[10, 30, 30])
#
#     sitk.WriteImage(cropped_img, '%s/%s.nii.gz'%(save_path, name))
#     sitk.WriteImage(cropped_lab, '%s/%s_gt.nii.gz'%(save_path, name))
#
#
# if __name__ == '__main__':
#
#
#     src_path = '/research/cbim/medical/yg397/LiTS/data/'
#     tgt_path = '/research/cbim/medical/yg397/tgt_dir/'
#
#
#     name_list = []
#     for i in range(0, 131):
#         name_list.append(i)
#
#     if not os.path.exists(tgt_path+'list'):
#         os.mkdir('%slist'%(tgt_path))
#     with open("%slist/dataset.yaml"%tgt_path, "w",encoding="utf-8") as f:
#         yaml.dump(name_list, f)
#
#     os.chdir(src_path)
#
#     for name in name_list:
#         img_name = 'volume-%d.nii'%name
#         lab_name = 'segmentation-%d.nii'%name
#
#         img = sitk.ReadImage(src_path+img_name)
#         lab = sitk.ReadImage(src_path+lab_name)
#
#         ResampleImage(img, lab, tgt_path, name, (0.767578125, 0.767578125, 1.0))
#         #ResampleImage(img, lab, tgt_path, name, (1, 1, 1))
#         print(name, 'done')
#

import numpy as np
import SimpleITK as sitk
from utils import ResampleXYZAxis, ResampleLabelToRef, CropForeground
import os
import random
import yaml
import copy
import pdb

def window_transform(ct_array, windowWidth, windowCenter, normal=False):
    """
    return: trucated image according to window center and window width
    and normalized to [0,1]
    """
    minWindow = float(windowCenter) - 0.5*float(windowWidth)
    newimg = (ct_array - minWindow) / float(windowWidth)
    shape=newimg.shape
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    if not normal:
        newimg = (newimg * 255).astype('float32')
    return newimg


def ResampleImage(imImage, imLabel, save_path, name, target_spacing=(1., 1., 1.)):
    assert imImage.GetSize() == imLabel.GetSize()

    spacing = imImage.GetSpacing()
    origin = imImage.GetOrigin()

    imLabel.CopyInformation(imImage)

    # npimg = sitk.GetArrayFromImage(imImage).astype(np.int32)
    # nplab = sitk.GetArrayFromImage(imLabel).astype(np.uint8)
    npimg = sitk.GetArrayFromImage(imImage)
    nplab = sitk.GetArrayFromImage(imLabel)
    z, y, x = npimg.shape

    if not os.path.exists('%s' % (save_path)):
        os.mkdir('%s' % (save_path))

    # imImage.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    # imLabel.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

    re_img_xy = ResampleXYZAxis(imImage, space=(target_spacing[0], target_spacing[1], target_spacing[2]),interp=sitk.sitkBSpline)
    re_lab_xy = ResampleLabelToRef(imLabel, re_img_xy, interp=sitk.sitkNearestNeighbor)

    # re_img_xyz = ResampleXYZAxis(re_img_xy, space=(target_spacing[0], target_spacing[1], target_spacing[2]),
    #                              interp=sitk.sitkNearestNeighbor)
    # re_lab_xyz = ResampleLabelToRef(re_lab_xy, re_img_xyz, interp=sitk.sitkNearestNeighbor)

    # cropped_img, cropped_lab = CropForeground(re_img_xyz, re_lab_xyz, context_size=[10, 30, 30])

    sitk.WriteImage(re_img_xy, '%s/%s.nii.gz' % (save_path, name))
    sitk.WriteImage(re_lab_xy, '%s/%s_gt.nii.gz' % (save_path, name))

if __name__ == '__main__':

    # src_path = '/opt/data/private/zjm/UTnetv2_new/dataset/lits_2d/'
    # tgt_path = '/opt/data/private/zjm/UTnetv2_new/dataset/lits_2d_new/'

    src_path = '/opt/data/private/zjm/UTnetV2/dataset/flare2021/imagesTr/'
    tgt_path = '/opt/data/private/zjm/UTnetV2/dataset/flare21_2d/'
    name_list = []
    for i in range(1, 362):
        name_list.append(i)

    if not os.path.exists(tgt_path + 'list'):
        os.mkdir('%slist' % (tgt_path))
    with open("%slist/dataset.yaml" % tgt_path, "w", encoding="utf-8") as f:
        yaml.dump(name_list, f)

    os.chdir(src_path)

    for name in name_list:
        # img_name = 'volume-%d.nii' % name
        # lab_name = 'segmentation-%d.nii' % name
        img_name = 'train_'+str(name).rjust(4,'0')+'_0000'
        lab_name = 'train_'+str(name).rjust(4,'0')

        img = sitk.ReadImage(src_path + img_name)
        lab = sitk.ReadImage(src_path + lab_name)
        array=sitk.GetArrayFromImage(img)
        array=window_transform(array,350,30,False)
        img=sitk.GetImageFromArray(array)
        ResampleImage(img, lab, tgt_path, name, (1.1428572, 1.1428572,1.5))
        # ResampleImage(img, lab, tgt_path, name, (1.2, 1.2, 1.2))
        print(name, 'done')



