import torch
from src.reftr.models.reftr import Reftr, SetCriterion
from src.reftr.models.transfomer import build_transformer


def build_model(args):
    # assertions
    assert args.num_steps > 2, "num_steps must be greater than 2."
    assert args.traj_train_len > 1, "traj_train_len must be greater than 1."
    if args.retain_past_lvls_num > 0:
        assert args.retain_past_lvls_num < args.traj_train_len, "retain_past_lvls_num should be less than traj_train_len!"

    class_dict = {'vessel': 0, 'background': 1}
    args.class_dict = class_dict
    device = torch.device(args.device)
    transformer = build_transformer(args)

    model = Reftr(
        transformer=transformer,
        num_prev_pos=args.num_prev_pos,
        num_prod_dec_layers=args.dec_prod_layers,
        num_dir_dec_layers=args.dec_dir_layers,
        seq_len=args.seq_len,
        num_init_branches=args.num_init_branches,
        traj_train_len=args.traj_train_len,
        div_mlp_depths=args.div_mlp_depths,
        end_mlp_depths=args.end_mlp_depths,
        mlp_activation=args.mlp_activation,
        retain_past_lvls_num=args.retain_past_lvls_num,
    )

    weight_dict = {'loss_divergence': args.div_loss_coef,
                   'loss_end': args.end_loss_coef,
                   'loss_direction': args.dir_loss_coef,
                   'loss_radius': args.rad_loss_coef, }
    losses = ['direction', 'radius', 'divergence', 'end']

    criterion = SetCriterion(
        losses=losses,
        num_init_branches=args.num_init_branches,
        cost_direction=args.set_cost_direction,
        cost_radius=args.set_cost_radius,
        train_div_non_bif_prob=args.train_div_non_bif_prob,
        train_div_bif_prob=args.train_div_bif_prob,
        weight_dict=weight_dict,
    )

    model.to(device)
    criterion.to(device)

    return model, criterion