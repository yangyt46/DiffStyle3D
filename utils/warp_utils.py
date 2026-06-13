import math
import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from PIL import Image
def camera_to_KT(camera):
    # === Build K ===
    W = camera.image_width
    H = camera.image_height

    fx = 0.5 * W / math.tan(camera.FoVx * 0.5)
    fy = 0.5 * H / math.tan(camera.FoVy * 0.5)
    cx = W * 0.5
    cy = H * 0.5

    K = torch.tensor([[fx, 0, cx],
                      [0, fy, cy],
                      [0,  0,  1]], device=camera.data_device).float()

    # === Build T ===
    T_c2w = camera.world_view_transform.inverse()  # (4,4)

    return K, T_c2w

def warp_batch_multiview(
    depths,    # (B, 1, H, W)
    Ks,        # (B, 3, 3)
    Ts,        # (B, 4, 4) camera-to-world
    eps=1e-4,
):
    """
    Returns:
        grids: (B, J, H, W, 2)   # grid[b, j] = warp view j → view b
        masks: (B, H, W)     # new_masks[b]
    """

    device = depths.device
    B, H, W = depths.shape

    # -------------------------------------------------
    # 1. pixel grid in reference view
    # -------------------------------------------------
    u, v = torch.meshgrid(
        torch.arange(W, device=device),
        torch.arange(H, device=device),
        indexing="xy",
    )
    pix = torch.stack([u, v, torch.ones_like(u)], dim=0)  # (3, H, W)
    pix = pix.reshape(3, -1).float()                      # (3, HW)

    # -------------------------------------------------
    # 2. backproject for ALL reference views
    # -------------------------------------------------
    Ks_inv = torch.inverse(Ks)                            # (B,3,3)
    depths_flat = depths.reshape(B, 1, -1)                # (B,1,HW)

    pts_cam = (Ks_inv @ pix[None]) * depths_flat           # (B,3,HW)
    pts_cam_h = torch.cat(
        [pts_cam, torch.ones_like(pts_cam[:, :1])], dim=1
    )                                                      # (B,4,HW)

    # -------------------------------------------------
    # 3. camera transforms
    # -------------------------------------------------
    T_c2w = Ts
    T_w2c = torch.inverse(Ts)

    # world coords for each reference view
    pts_world = T_c2w @ pts_cam_h                          # (B,4,HW)

    # -------------------------------------------------
    # 4. project into ALL source views
    # -------------------------------------------------
    # expand for pairwise (b, j)
    pts_world = pts_world[:, None].expand(B, B, 4, -1)     # (B,B,4,HW)
    T_w2c_j = T_w2c[None].expand(B, B, 4, 4)                # (B,B,4,4)
    Ks_j = Ks[None].expand(B, B, 3, 3)                      # (B,B,3,3)

    pts_cam_j = (T_w2c_j @ pts_world)[..., :3, :]           # (B,B,3,HW)

    proj = Ks_j @ pts_cam_j                                 # (B,B,3,HW)

    z = proj[..., 2, :]                                     # (B,B,HW)
    u2 = proj[..., 0, :] / (z + eps)
    v2 = proj[..., 1, :] / (z + eps)

    # -------------------------------------------------
    # 5. build sampling grid
    # -------------------------------------------------
    grid_x = 2.0 * (u2 / (W - 1)) - 1.0
    grid_y = 2.0 * (v2 / (H - 1)) - 1.0

    grids = torch.stack([grid_x, grid_y], dim=-1)           # (B,B,HW,2)
    grids = grids.view(B, B, H, W, 2)

    # -------------------------------------------------
    # 6. geometric visibility (bool)
    # -------------------------------------------------
    in_front = z > eps
    in_bound = (
            (grid_x >= -1.0) & (grid_x <= 1.0) &
            (grid_y >= -1.0) & (grid_y <= 1.0)
    )

    visible = (in_front & in_bound).view(B, B, H, W)
    for b in range(B):
        visible[b, b].fill_(True)
    # -------------------------------------------------
    # 7. temporal new-region mask
    # -------------------------------------------------
    new_masks = torch.zeros(B, H, W, device=device)
    for b in range(B):
        if b == 0:
            new_masks[b].fill_(1.0)
        else:
            seen_before = visible[:b, b].any(dim=0)
            new_masks[b] = (~seen_before).float()
    return grids, new_masks,visible
def warp_batch(rgbs, depths, Ks, Ts, ref_id=0, eps=1e-4):

    device = rgbs.device
    B, _, H, W = rgbs.size()

    # ----------------------
    # 1. Reference pixel grid
    # ----------------------
    u, v = torch.meshgrid(
        torch.arange(W, device=device),
        torch.arange(H, device=device),
        indexing='xy'
    )
    pix = torch.stack([u, v, torch.ones_like(u)], dim=0).float().reshape(3, -1)

    # ----------------------
    # 2. Backproject from ref view
    # ----------------------
    D0 = depths[ref_id].reshape(1, -1)
    K0_inv = torch.inverse(Ks[ref_id])
    pts_cam0 = (K0_inv @ pix) * D0                # (3,HW)
    pts_cam0_h = torch.cat([pts_cam0, torch.ones_like(pts_cam0[:1])], dim=0)  # (4,HW)

    # ----------------------
    # 3. Camera transforms
    # Ts assumed to be camera-to-world (c2w)
    # ----------------------
    T_c2w = Ts
    T_w2c = torch.inverse(Ts)

    # 3. Transform into world coords
    pts_world = T_c2w[ref_id] @ pts_cam0_h    # (4,HW)
    pts_world = pts_world.unsqueeze(0).expand(B, 4, -1)

    # ----------------------
    # 4. Project into each camera
    # ----------------------
    pts_cam = (T_w2c @ pts_world)[:, :3]  # (B,3,HW)
    proj = Ks @ pts_cam
    u2 = proj[:, 0] / (proj[:, 2] + eps)
    v2 = proj[:, 1] / (proj[:, 2] + eps)

    # ----------------------
    # 5. Build sampling grid
    # ----------------------
    grid_x = 2 * (u2 / (W - 1)) - 1
    grid_y = 2 * (v2 / (H - 1)) - 1
    grid = torch.stack([grid_x, grid_y], dim=-1).reshape(B, H, W, 2)

    # ----------------------
    # 6. Warp rgb
    # ----------------------
    I_warp = F.grid_sample(rgbs, grid, align_corners=True)

    # ----------------------
    # 7. Occlusion mask
    # (Z_proj must be in front of surface in the target view)
    # ----------------------
    Z_proj = pts_cam[:, 2].reshape(B, H, W)
    Z_true = depths

    mask = (Z_proj <= Z_true + 0.01).float()   # valid if not occluded

    return I_warp, grid, mask
def _tensor_size(t):
    return t.size()[1]*t.size()[2]*t.size()[3]
def sh_rest_to_gray(f_rest):
    """
    f_rest: [N, 3*K]
     SH basis 有 (R_k, G_k, B_k)
    """
    C = f_rest.shape[1]
    assert C % 3 == 0, "features_rest must be RGB-blocked SH coefficients"

    K = C // 3
    r = f_rest[:, 0:K]
    g = f_rest[:, K:2*K]
    b = f_rest[:, 2*K:3*K]

    gray = 0.299 * r + 0.587 * g + 0.114 * b

    return torch.cat([gray, gray, gray], dim=1)

def sh_dc_init_gray(f_dc):
    """
    f_dc: [N, 1, 3]  (DC SH × RGB)
    """
    rgb = f_dc[:, 0, :]  # [N, 3]

    gray = (
        0.299 * rgb[:, 0] +
        0.587 * rgb[:, 1] +
        0.114 * rgb[:, 2]
    )  # [N]

    gray_rgb = gray.unsqueeze(1).repeat(1, 3)  # [N, 3]

    return gray_rgb.unsqueeze(1)  # [N, 1, 3]

def load_image(image_path, size=None, mode="RGB"):
    img = Image.open(image_path).convert(mode)
    if size is None:
        width, height = img.size
        new_width = (width // 64) * 64
        new_height = (height // 64) * 64
        size = (new_width, new_height)
    img = img.resize(size, Image.BICUBIC)
    return ToTensor()(img).unsqueeze(0)