"""
这段代码主要用于处理3D渲染图像数据。以下是其主要内容：
定义了一些变换矩阵，这些矩阵用于在3D空间中移动和旋转对象：
trans_t(t)：生成3D平移矩阵，用于在z轴上移动物体。
rot_phi(phi)和rot_theta(th)：生成3D旋转矩阵，用于在特定角度旋转物体。
pose_spherical(theta, phi, radius)：
这是一个生成球坐标变换矩阵的函数。首先将物体在z轴上移动指定的半径，然后通过给定的角度进行转动。最后，通过一个固定的旋转矩阵进行一次旋转。
load_blender_data(basedir, half_res=False, testskip=1)：
这个函数是加载Blender渲染的图像数据，并按训练、验证和测试数据划分。
首先，它将从给定的路径加载多个JSON文件，这些JSON文件包含了渲染每一帧时的变换矩阵和相机参数等元数据。
对每一帧，合并图像数据并载入对应的变换矩阵。
合并所有图像，并且对图像进行了规范化处理（由255值范围归一化至0-1范围）。
如果 half_res 参数为 True，则将图像缩小至原始大小的一半。
计算并返回相机的焦距。
计算40个不同角度的旋转矩阵，并将它们存储到 render_poses 变量中，这些矩阵可以用于渲染新的图像。

"""


import os
import tensorflow as tf
import numpy as np
import imageio 
import json




trans_t = lambda t : tf.convert_to_tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1],
], dtype=tf.float32)

rot_phi = lambda phi : tf.convert_to_tensor([
    [1,0,0,0],
    [0,tf.cos(phi),-tf.sin(phi),0],
    [0,tf.sin(phi), tf.cos(phi),0],
    [0,0,0,1],
], dtype=tf.float32)

rot_theta = lambda th : tf.convert_to_tensor([
    [tf.cos(th),0,-tf.sin(th),0],
    [0,1,0,0],
    [tf.sin(th),0, tf.cos(th),0],
    [0,0,0,1],
], dtype=tf.float32)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w
    


def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    render_poses = tf.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]],0)
    
    if half_res:
        imgs = tf.image.resize_area(imgs, [400, 400]).numpy()
        H = H//2
        W = W//2
        focal = focal/2.
        
    return imgs, poses, render_poses, [H, W, focal], i_split


