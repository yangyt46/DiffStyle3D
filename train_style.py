from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
import os
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams,GuidanceParams
from guidance.AttnGuidance import StyleDiffusion
from utils.warp_utils import *
import torch.nn.functional as F

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim

    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam

    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe,guidance_opt, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(
            f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset,opt, pipe,guidance_opt)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians,shuffle=False)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        first_iter = 0

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE

    with torch.no_grad():
        gaussians._features_rest[:] = sh_rest_to_gray(gaussians._features_rest)
        gaussians._features_dc[:] = sh_dc_init_gray(gaussians._features_dc)
    gaussians._xyz.requires_grad_(False)
    gaussians._scaling.requires_grad_(False)
    gaussians._rotation.requires_grad_(False)
    gaussians._opacity.requires_grad_(False)
    l = [
        {'params': [gaussians._features_dc], 'lr': args.feature_lr, "name": "f_dc"},
        {'params': [gaussians._features_rest], 'lr': args.feature_lr/120.0, "name": "f_rest"},
    ]
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    #
    gaussians.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    ############################ stylediffusion
    style_image = load_image(guidance_opt.style_image_path, size=(guidance_opt.height, guidance_opt.width))
    guidance = StyleDiffusion(device=guidance_opt.g_device,style_image=style_image,
                              start_layer=guidance_opt.start_layer,end_layer=guidance_opt.end_layer,
                              height=guidance_opt.height,width=guidance_opt.width,
                              content_weight=guidance_opt.content_weight,
                              num_inference_steps=guidance_opt.num_inference_steps,
                              guidance_opt=guidance_opt)

    ###################
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    total_viewpoint_stack = scene.getTrainCameras().copy()
    cp = len(total_viewpoint_stack) % guidance_opt.batch_size
    if cp != 0:
        total_viewpoint_stack = total_viewpoint_stack + total_viewpoint_stack[len(total_viewpoint_stack) - guidance_opt.batch_size + cp:]

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer,
                                       use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)[
                        "render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()
        gaussians.update_learning_rate(iteration)


        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = total_viewpoint_stack.copy()

        images = []
        gt_images = []
        ks = []
        ts = []
        depth_images =[]
        for i in range(guidance_opt.batch_size):
            try:
                viewpoint_cam = viewpoint_stack.pop(0)
            except:
                viewpoint_stack = total_viewpoint_stack.copy()
                viewpoint_cam = viewpoint_stack.pop(0)
            gt_image = viewpoint_cam.original_image.cuda(non_blocking=True)
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            image, viewspace_point_tensor, visibility_filter, radii, depth_image= render_pkg["render"], render_pkg["viewspace_points"], \
            render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["depth"]
            images.append(image)
            gt_images.append(gt_image)
            depth_images.append(depth_image.squeeze())
            K,T = camera_to_KT(viewpoint_cam)
            ks.append(K)
            ts.append(T)
        images = torch.stack(images, dim=0)
        gt_images = torch.stack(gt_images, dim=0)
        depth_images = torch.stack(depth_images, dim=0)
        ks = torch.stack(ks, dim=0)
        ts = torch.stack(ts, dim=0)
        grid, mask, visible= warp_batch_multiview(depth_images,ks,ts)

        images_pred = F.interpolate(images, size=(guidance_opt.height, guidance_opt.width),
                                   mode="bilinear", align_corners=True)
        gt_pred = F.interpolate(gt_images, size=(guidance_opt.height, guidance_opt.width),
                                   mode="bilinear", align_corners=True)
        loss = guidance.train_style(latents=images_pred,gt_content=gt_pred,iteration=iteration,grid=grid ,visible=visible.float(),mask=mask)
        loss.backward()
        iter_end.record()

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, 0, loss, l1_loss, iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render,
                            (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp),
                            dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            if (iteration < guidance_opt.end_reset_iter) and (iteration > guidance_opt.start_reset_iter):
                qs = torch.zeros((guidance_opt.batch_size))
                for i in range(guidance_opt.batch_size):
                    image_hsv = images[i].mean(0)
                    image_mask = image_hsv[image_hsv > 0.]
                    if len(image_mask) > 0:
                        qs[i]= torch.quantile(image_mask, q=guidance_opt.thresh, dim=-1)
                    else:
                        qs[i]= 0.
                qs = torch.quantile(qs, q=guidance_opt.thresh)
                if qs > guidance_opt.lgt_thresh:
                    gaussians.reset_brightness(guidance_opt.reset_rate)

            # Optimizer step
            if iteration < opt.iterations:
                #rewrite
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none=True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(dataset,opt, pipe,guidance_opt):
    if not dataset.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        dataset.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(dataset.model_path))
    os.makedirs(dataset.model_path, exist_ok=True)
    with open(os.path.join(dataset.model_path, "cfg_args"), 'w') as cfg_log_f:
        # cfg_log_f.write(str(Namespace(**vars(dataset))))
        dump_namespace(cfg_log_f, "DATASET", dataset)
        dump_namespace(cfg_log_f, "OPT", opt)
        dump_namespace(cfg_log_f, "GUIDANCE_OPT", guidance_opt)

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(dataset.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def dump_namespace(f, title, ns):
    f.write(f"===== {title} =====\n")
    items = sorted(vars(ns).items())
    max_len = max(len(k) for k, _ in items)
    for k, v in items:
        f.write(f"{k.ljust(max_len)} : {v}\n")
    f.write("\n")

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, train_test_exp):
    if tb_writer:
        # tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1, iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    gp = GuidanceParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, required=True,
                        help="please provide the starting checkpoint file")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)

    args.checkpoint_iterations.append(args.iterations)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args),gp.extract(args), args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
