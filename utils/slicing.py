############################获取投影###########################################
import SimpleITK as sitk
import numpy as np
from PIL import Image

def windowAdjust(img, ww, wl):
    win_min = wl - ww / 2
    win_max = wl + ww / 2
    img_new = np.clip(img, win_min, win_max)
    return img_new

def Normalization(hu_value):
    hu_min = np.min(hu_value)
    hu_max = np.max(hu_value)
    normal_value = (hu_value - hu_min) / (hu_max - hu_min)
    return normal_value

def get_photo_mat(px, py, pz, space_origin, pixel_spacing, sample_num=256):
    # space_origin = p0_lps + i * delta * pz - sample_num / 2 * px - sample_num / 2 * py - 0 * pz
    '''
    extrinsic, intrinsic: world2plane
    extrinsic_, intrinsic_: plane2world
    '''

    extrinsic_ = np.identity(4)
    extrinsic_[0:3, 0] = px
    extrinsic_[0:3, 1] = py
    extrinsic_[0:3, 2] = pz
    extrinsic_[0:3, 3] = space_origin # TODO: 手工定义的R^3向量
    extrinsic = np.identity(4)
    extrinsic[0:3, 0:3] = np.linalg.inv(extrinsic_[0:3, 0:3])
    extrinsic[0:3, 3] = -np.dot(np.linalg.inv(extrinsic_[0:3, 0:3]), space_origin)

    intrinsic = np.identity(4)
    fx = 1 / pixel_spacing[0]
    fy = 1 / pixel_spacing[1]
    intrinsic[0, 0] = fx
    intrinsic[1, 1] = fy
    intrinsic[2, 2] = 0
    # Assume that space_origin is at the center of the pic
    u0v0 = np.dot(extrinsic, np.array([space_origin[0], space_origin[1], space_origin[2], 1]))
    u0 = sample_num / 2 - fx * u0v0[0]
    v0 = sample_num / 2 - fy * u0v0[1]
    intrinsic[0, 3] = u0
    intrinsic[1, 3] = v0

    intrinsic_ = np.identity(4)
    intrinsic_[0, 0] = pixel_spacing[0]
    intrinsic_[1, 1] = pixel_spacing[1]
    intrinsic_[2, 2] = 0
    intrinsic_[0, 3] = -pixel_spacing[0] * u0
    intrinsic_[1, 3] = -pixel_spacing[1] * v0
    # intrinsic_[0:3, 0:3] = np.linalg.inv(intrinsic[0:3, 0:3])
    # intrinsic_[0:3, 3] = -np.dot(np.linalg.inv(intrinsic[0:3, 0:3]), intrinsic[0:3, 3])

    return extrinsic, intrinsic, extrinsic_, intrinsic_



def slicing(path_volume):
    def lerp(lo, hi, t):
        return lo * (1 - t) + hi * t

    def LERP(map, bg_map, mat):
        for i in range(map.shape[0]):
            for j in range(map.shape[1]):
                point_lps = np.dot(mat, np.array([i, j, 0, 1]))
                x = int(point_lps[0])
                y = int(point_lps[1])
                z = int(point_lps[2])
                tx = point_lps[0] - x
                ty = point_lps[1] - y
                tz = point_lps[2] - z
                if (x < 0 or x > bg_map.shape[0] - 2 or y < 0 or y > bg_map.shape[1] - 2 or z < 0 or z > bg_map.shape[
                    2] - 2):
                    continue
                c000 = bg_map[x, y, z]
                c100 = bg_map[x + 1, y, z]
                c010 = bg_map[x, y + 1, z]
                c110 = bg_map[x + 1, y + 1, z]
                c001 = bg_map[x, y, z + 1]
                c101 = bg_map[x + 1, y, z + 1]
                c011 = bg_map[x, y + 1, z + 1]
                c111 = bg_map[x + 1, y + 1, z + 1]

                a = lerp(c000, c100, tx)
                b = lerp(c010, c110, tx)
                c = lerp(c001, c101, tx)
                d = lerp(c011, c111, tx)
                e = lerp(a, b, ty)
                f = lerp(c, d, ty)
                p_lerp = lerp(e, f, tz)

                map[i, j] = p_lerp

    def slice(path_volume):
        #### get volume information
        volume = sitk.ReadImage(path_volume)
        npy_volume = sitk.GetArrayFromImage(volume)  # z,y,x
        npy_volume = npy_volume.transpose(2, 1, 0)  # x,y,z

        space_origin = np.array(volume.GetOrigin())
        space_directions = np.array(volume.GetDirection())
        space_spacing = np.array(volume.GetSpacing())
        # print(space_origin,space_directions,space_spacing) [ -45.5         228.58453369 -271.88000488] [ 1.  0.  0.  0. -1.  0.  0.  0.  1.] [0.35546875 0.35546875 0.44999999]
        space_directions = space_directions.reshape(3, 3, order="c")
        space_directions = space_spacing * space_directions # TODO: 单位坐标 ?

        img2world = np.identity(4)
        img2world[0:3, 0:3] = space_directions
        img2world[0:3, 3] = space_origin

        world2image = np.identity(4)
        world2image[0:3, 0:3] = np.linalg.inv(space_directions)
        world2image[0:3, 3] = -np.dot(np.linalg.inv(space_directions), space_origin)

        #### get slice information
        # plane function: 给定投影平面的x,y,z轴向量,中心坐标,像素间距
        # TODO: 建立正交坐标系 ?
        sample_num = 512
        px = np.array([0, 1, 1])
        px = px / np.linalg.norm(px)
        pyy = np.array([0, 0, -1])
        pz = np.cross(px, pyy)
        pz = pz / np.linalg.norm(pz)
        py = np.cross(pz, px)
        py = py / np.linalg.norm(py)
        # print(px,py,pz) [0.         0.70710678 0.70710678] [ 0.          0.70710678 -0.70710678] [-1.  0.  0.]

        plane_origin = np.array([45, 0, -164])  # Assume that plane_origin is at the center of the pic
        extrinsic, intrinsic, extrinsic_, intrinsic_ = get_photo_mat(px, py, pz, plane_origin, space_spacing,
                                                                     sample_num)

        '''
            Note:
            plane2image = world2image * plane2world
            plane2world = world2plane.inv = (intrinsic * extrinsic).inv = extrinsic_inv * intrinsic_inv
        '''
        plane2world = np.dot(extrinsic_, intrinsic_)  # [4, 4]
        plane2image = np.dot(world2image, plane2world)

        #### Slicing
        map = np.zeros([sample_num, sample_num])
        LERP(map, npy_volume, plane2image)
        img_slice = Normalization(windowAdjust(map, 800, 200)) * 255  # 调节窗位窗宽
        img_slice = np.array(img_slice, dtype=np.uint8).transpose()
        image = Image.fromarray(img_slice)
        image.save("./ct_train_1001_image.png")
    slice(path_volume)


if __name__ == "__main__":
    path_volume = r"/home/Bigdata/medical_dataset/MM-WHS/MM-WHS/CT/train/ct_train_1001_image.nii.gz"
    slicing(path_volume)





