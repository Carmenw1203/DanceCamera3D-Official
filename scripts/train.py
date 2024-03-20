from args import parse_train_opt
from DanceCamera3D import DanceCamera3D

def train(opt):
    if opt.backbone == 'diffusion':
        model = DanceCamera3D(feature_type = opt.feature_type,
                            learning_rate = opt.learning_rate,
                            camera_format = opt.camera_format,
                            condition_separation_CFG = opt.condition_separation_CFG,
                            w_loss = opt.w_loss,
                            w_v_loss = opt.w_v_loss,
                            w_a_loss = opt.w_a_loss,
                            w_in_ba_loss = opt.w_in_ba_loss,
                            w_out_ba_loss = opt.w_out_ba_loss,)
    model.train_loop(opt)


if __name__ == "__main__":
    opt = parse_train_opt()
    train(opt)