import json
import os.path as osp
if __name__ == '__main__':
    annot_path = '/mnt/data1/tyluan/workspace/dataset/InterHand2.6M_5fps_batch1/annotations'
    modes = ['test', 'train', 'val']
    cam_list = {}
    for mode in modes:
        with open(osp.join(annot_path, mode, 'InterHand2.6M_' + mode + '_joint_3d.json')) as f:
            joints = json.load(f)
        for sub in joints:
            for cam in joints[sub]:
                if cam not in cam_list:
                    cam_list[cam] = [sub+mode]
                else:
                    cam_list[cam].append(sub+mode)
    image_list = []
    for img in cam_list:
        image_list.append(int(img))
    # for img in sorted(image_list):
    #     if img < 10000:
    #         print(img, cam_list[str(img)])

    dhm_path = '/mnt/data1/tyluan/workspace/dataset/DeepHandMesh/annotations/keypoints/subject_4'
    import os
    dhm_list = os.listdir(dhm_path)
    for filename in dhm_list:
        img = filename[9:14]
        if img[-1] == '.':
            img = img[:-1]
            img_int = int(img)
            
            if img_int in image_list:
                print(img_int, cam_list[str(img_int)]) 
