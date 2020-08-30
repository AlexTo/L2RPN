import argparse


def args_parser():
    """some default command line arguments (cli) for training the baselines. Can be reused in some baselines here."""
    parser = argparse.ArgumentParser(description="Train BeU agent")
    parser.add_argument("--num_train_steps", required=False,
                        default=1024, type=int,
                        help="Number of training iterations")
    parser.add_argument("--save_path", required=False,
                        help="Path where the model should be saved.")
    parser.add_argument("--name", required=False,
                        help="Name given to your model.")
    parser.add_argument("--load_path", required=False,
                        help="Path from which to reload your model from (by default ``None`` to NOT reload anything)")
    parser.add_argument("--env_name", required=False,
                        # default="l2rpn_neurips_2020_track2_small",
                        default="rte_case14_realistic",
                        help="Name of the environment to load (default \"rte_case14_realistic\"")
    parser.add_argument("--logs_dir", required=False, default=None,
                        help="Where to output the training logs (usually tensorboard logs)")

    parser.add_argument("--lr", type=float, default=0.00002)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-06)

    parser.add_argument("--feature_list", required=False,
                        default="day_of_week,hour_of_day,minute_of_hour,prod_p,prod_v,load_p,load_q,actual_dispatch,"
                                "target_dispatch,topo_vect,time_before_cooldown_line,time_before_cooldown_sub,rho,"
                                "timestep_overflow,line_status",
                        help="List of considered features")
    return parser
