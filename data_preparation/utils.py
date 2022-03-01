import numpy as np

def cal_patchgt_score(input_mask, patch_gt, patch_score, patch_class, patch_size, delta=0.1):
    """calculate patch ground truth and score from mask

    Args:
        input_mask (np.narray): real or fake mask
        patch_gt (list): patch ground truth
        patch_score (list): patch score, for measuring the importance of patches
        patch_class (int): the patch class of input_mask
        patch_size (int): the size of each patch
        delta (int, optional): a threshold value, prevent mislabeling due to small area ratio. Defaults to 0.1

    Returns:
        patch_gt (list): patch ground truth
        patch_score (list): patch score
    """
    w, h = input_mask.shape
    row = int(h / patch_size)  # row of patch
    col = int(w / patch_size)  # col of patch
    for r in range(row):
        for c in range(col):
            patch = input_mask[r * patch_size : r * patch_size + patch_size, c * patch_size : c * patch_size + patch_size]
            ratio = float(patch.sum()) / float(patch_size * patch_size)  # calculate area ratio of this patch_class
            patch_gt[r * col + c] = patch_class if ratio >= delta else patch_gt[r * col + c]  # check whether the area ratio is greater than or equal to delta. If not, keep the patch_gt unchanged
            patch_score[r * col + c] = 2.0 - ratio if ratio >= delta else patch_score[r * col + c]  # get patch score
    return patch_gt, patch_score

def hop2():
    """patch pair mode, here we use hop-2

    Returns:
        off_rs (list): which row should be connected
        off_cs (list): which col should be connected
    """
    off_rs = range(0,3)
    off_cs = range(-2,3)
    return off_rs, off_cs

def get_patch_pair(input_image_size:int, patch_size:int, patch_pair_mode="hop2"):
    """Patch pairs used in the loss

    Args:
        input_image_size (int): the size of input image.
        patch_size (int): patch size.

    Returns:
        patch_pair (list): [[patch_{start},patch_{end}], ..., [patch_{start},patch_{end}]]
    """
    row, col = [int(input_image_size / patch_size)] * 2  # the row and col of patches
    off_rs, off_cs = eval(patch_pair_mode)()  # get patch pair mode
    patch_pair = []  # [[patch_{start},patch_{end}], ..., [patch_{start},patch_{end}]]

    for r in range(row):
        for c in range(col):
            for off_r in off_rs:
                for off_c in off_cs:
                    # remove over-the-boundary
                    if r + off_r < 0 or r + off_r >= row or c + off_c < 0 or c + off_c >= col:
                        continue
                    # remove the symmetry
                    if [r * col + c, (r + off_r) * col + (c + off_c)] in patch_pair or [(r + off_r) * col + (c + off_c), r * col + c] in patch_pair:
                        continue
                    # remove self-connection
                    if r * col + c == (r + off_r) * col + (c + off_c):
                        continue
                    patch_pair.append([r * col + c, (r + off_r) * col + (c + off_c)])
    return patch_pair

def convert_patchpair_format(patch_pair):
    """convert patch pair format

    Args:
        patch_pair (list): [[patch_{start},patch_{end}], ..., [patch_{start},patch_{end}]]

    Returns:
        [list]: [src_patch_list, tgt_patch_list], such as [[1,2,3],[4,5,6]]
    """
    patch_pair = np.array(patch_pair)
    return np.array([patch_pair[:,0], patch_pair[:,1]])