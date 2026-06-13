from argparse import ArgumentParser, Namespace
import sys
import os
import ast

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class GuidanceParams(ParamGroup):
    def __init__(self, parser):
        self.g_device = "cuda"
        self.sd_model_key = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
        self.num_inference_steps=1000
        self.start_layer = 0
        self.end_layer = 16
        self.style_image_path = ""
        self.height=512
        self.width=512
        self.content_weight=0.2
        self.max_iteration = 10000
        self.grad_clip = [2.0, 8.0, 1000]
        self.start_reset_iter = 0
        self.end_reset_iter   = 300
        self.thresh = 0.85
        self.lgt_thresh = 0.9
        self.reset_rate =  0.8
        self.batch_size = 4
        self.svd_vae = False
        self.set_timestep = 0
        self.add_noise=False
        self.fp16 = False
        super().__init__(parser, "Guidance Model Parameters")

class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        # self.feature_lr = 0.0025
        self.feature_lr = 0.02
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        self.optimizer_type = "default"
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

def get_combined_yaml_args(parser: ArgumentParser):
    cmdlne_args = sys.argv[1:]
    args_cmdline = parser.parse_args(cmdlne_args)

    # 默认空 Namespace
    args_cfgfile = Namespace()

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath, "r") as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            lines = cfg_file.readlines()

        # 解析 cfg 文件
        cfg_dict = {}
        current_section = None
        target_section = "DATASET"  # 只取 DATASET 分组

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("=====") and line.endswith("====="):
                section_name = line.strip("=").strip()
                current_section = section_name
                cfg_dict[current_section] = {}
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value == "":
                value_parsed = ""
            else:
                try:
                    value_parsed = ast.literal_eval(value)
                except:
                    value_parsed = value
            cfg_dict[current_section][key] = value_parsed

        # 生成 Namespace
        if target_section in cfg_dict:
            args_cfgfile = Namespace(**cfg_dict[target_section])
        else:
            print(f"Warning: section {target_section} not found in cfg file")
            args_cfgfile = Namespace()

    except FileNotFoundError:
        print("Config file not found at", cfgfilepath)
        pass

    # 合并命令行参数（优先覆盖）
    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v is not None:
            merged_dict[k] = v

    return Namespace(**merged_dict)