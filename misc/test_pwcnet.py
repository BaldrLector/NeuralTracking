from model.pwcnet import PWCNet
import torch
import numpy as np
from PIL import Image

saved_model = r'E:\workplace\CV\Reconstrction\RGBD_Reconstruct\NeuralTracking\flow_net.ckp'
pretrained_dict = torch.load(saved_model)
keys = list(pretrained_dict.keys())
for i in keys:
    v = pretrained_dict.pop(i)
    pretrained_dict[i[9:]] = v
flow_net = PWCNet().cuda()
flow_net.load_state_dict(pretrained_dict)

src_img = r'E:\database\vox\image\train\__jNzN4zM7A\000224#000553\0.jpg'
dri_img = r'E:\database\vox\image\train\__jNzN4zM7A\000224#000553\26.jpg'
src_img = np.array(Image.open(src_img))
dri_img = np.array(Image.open(dri_img))
src_img = torch.from_numpy(src_img).unsqueeze(
    0).permute([0, 3, 1, 2]).cuda().contiguous() / 255.
dri_img = torch.from_numpy(dri_img).unsqueeze(
    0).permute([0, 3, 1, 2]).cuda().contiguous() / 255.
image_height, image_width = src_img.shape[-2:]
batch_size = src_img.shape[0]

with torch.no_grad():
    flow2, flow3, flow4, flow5, flow6, features2 = flow_net.forward(
        src_img, dri_img)
    flow = -20.0 * torch.nn.functional.interpolate(input=flow2, size=(
        image_height, image_width), mode='bilinear', align_corners=False)

    x_coords = torch.arange(image_width, dtype=torch.float32, device=src_img.device).unsqueeze(
        0).expand(image_height, image_width).unsqueeze(0)
    y_coords = torch.arange(image_height, dtype=torch.float32, device=src_img.device).unsqueeze(
        1).expand(image_height, image_width).unsqueeze(0)

    xy_coords = torch.cat([x_coords, y_coords], 0)
    xy_coords = xy_coords.unsqueeze(0).repeat(
        batch_size, 1, 1, 1)  # (bs, 2, 448, 640)

    # Apply the flow to pixel coordinates.
    xy_coords_warped = xy_coords + flow
    xy_pixels_warped = xy_coords_warped.clone()

    # Normalize to be between -1, and 1.
    # Since we use "align_corners=False", the boundaries of corner pixels
    # are -1 and 1, not their centers.
    xy_coords_warped[:, 0, :, :] = (
        xy_coords_warped[:, 0, :, :]) / (image_width - 1)
    xy_coords_warped[:, 1, :, :] = (
        xy_coords_warped[:, 1, :, :]) / (image_height - 1)
    xy_coords_warped = xy_coords_warped * 2 - 1

    # Permute the warped coordinates to fit the grid_sample format.
    xy_coords_warped = xy_coords_warped.permute(0, 2, 3, 1)

    warped_src = torch.nn.functional.grid_sample(
        src_img, xy_coords_warped, mode="bilinear", padding_mode='zeros', align_corners=False
    )

    src_img = src_img.squeeze().permute([1, 2, 0]).cpu().numpy()
    dri_img = dri_img.squeeze().permute([1, 2, 0]).cpu().numpy()
    warped_src = warped_src.squeeze().permute([1, 2, 0]).cpu().numpy()
    all_img = np.concatenate([src_img, dri_img, warped_src], axis=1)*255
    Image.fromarray(all_img.astype(np.uint8)).show()
