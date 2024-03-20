import argparse


def parse_train_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="runs/train", help="project/name")
    parser.add_argument("--backbone",default="diffusion", help="backbone")
    parser.add_argument("--exp_name", default="exp", help="save to project/name")
    parser.add_argument("--data_path", type=str, default="data/amc_data_split_by_categories", help="raw data path")
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="data/dataset_backups/",
        help="Dataset backup path",
    )
    parser.add_argument(
        "--render_dir", type=str, default="renders/", help="Sample render path"
    )
    parser.add_argument(
        "--render_videos", type=bool, default=True, help="whether to render videos"
    )

    parser.add_argument("--feature_type", type=str, default="jukebox")
    parser.add_argument(
        "--wandb_pj_name", type=str, default="DanceCamera3D", help="project name"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument(
        "--force_reload", action="store_true", help="force reloads the datasets"
    )
    parser.add_argument(
        "--no_cache", action="store_true", help="don't reuse / cache loaded dataset"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=100,
        help='Log model after every "save_period" epoch',
    )
    parser.add_argument("--ema_interval", type=int, default=1, help="ema every x steps")
    parser.add_argument(
        "--checkpoint", type=str, default="", help="trained checkpoint path (optional)"
    )
    parser.add_argument("--learning_rate", type=float, default=0.0002, help="learning rate")
    parser.add_argument(
        "--camera_format",
        type=str,
        default="polar",# or centric
        help="polar coordinates or centric",
    )
    parser.add_argument(
        "--condition_separation_CFG",
        type=bool,
        default=True,# or centric
        help="conduct condition separation on CFG",
    )
    parser.add_argument(
        "--w_loss",
        type=float,
        default=2,
        help="loss weight",
    )
    parser.add_argument(
        "--w_v_loss",
        type=float,
        default=5,
        help="velocity loss weight",
    )
    parser.add_argument(
        "--w_a_loss",
        type=float,
        default=5,
        help="acceleration loss weight",
    )
    parser.add_argument(
        "--w_in_ba_loss",
        type=float,
        default=0.0015,
        help="inside body parts attention loss weight",
    )
    parser.add_argument(
        "--w_out_ba_loss",
        type=float,
        default=0,
        help="outside body parts attention loss weight",
    )
    opt = parser.parse_args()
    return opt


def parse_test_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_mode", type=str, default="normal")
    parser.add_argument("--backbone",default="diffusion", help="backbone")
    parser.add_argument("--feature_type", type=str, default="jukebox")
    parser.add_argument("--out_length", type=float, default=30, help="max. length of output, in seconds")
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="DCM_data/dataset_backups/",
        help="Dataset backup path",
    )
    parser.add_argument(
        "--render_dir", type=str, default="renders/test/", help="Sample render path"
    )
    parser.add_argument(
        "--label", type=str, default="exp-3000", help="Sample render path"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="train-3000.pt", help="checkpoint"
    )
    parser.add_argument(
        "--music_dir",
        type=str,
        default="DCM_data/amc_data_split_by_categories/Test/Audio",
        help="folder containing input music",
    )
    parser.add_argument(
        "--motionGT_dir",
        type=str,
        default="DCM_data/amc_data_split_by_categories/Test/Simplified_MotionGlobalTransform",
        help="folder containing Simplified_MotionGlobalTransform",
    )
    parser.add_argument(
        "--camera_dir",
        type=str,
        default="DCM_data/amc_data_split_by_categories/Test/CameraCentric",
        help="folder containing CameraCentric",
    )
    parser.add_argument(
        "--cache_features",
        action="store_true",
        help="Save the jukebox features for later reuse",
    )
    parser.add_argument(
        "--render_videos", action="store_true", help="whether to render videos"
    )
    parser.add_argument(
        "--use_cached_features",
        action="store_true",
        help="Use precomputed features instead of music folder",
    )
    parser.add_argument(
        "--feature_cache_dir",
        type=str,
        default="DCM_data/cached_features/",
        help="Where to save/load the features",
    )
    parser.add_argument(
        "--gw1",
        type=float,
        default=1.75,
        help="guidance weight 1",
    )
    parser.add_argument(
        "--gw2",
        type=float,
        default=1,
        help="guidance weight 2",
    )
    parser.add_argument(
        "--camera_format",
        type=str,
        default="polar",# or centric
        help="polar coordinates or centric",
    )
    parser.add_argument(
        "--condition_separation_CFG",
        type=bool,
        default=True,# or centric
        help="conduct condition separation on CFG",
    )
    opt = parser.parse_args()
    return opt
