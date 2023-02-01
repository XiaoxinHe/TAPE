import argparse


class BaseOptions:
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""

    def initialize(self):
        parser = argparse.ArgumentParser(description="Constrained learing")

        parser.add_argument(
            "--debug_mem_speed",
            action="store_true",
            help="whether to get the memory usage and throughput",
        )
        parser.add_argument("--debug", action="store_true")
        parser.add_argument("--tosparse", action="store_true")
        parser.add_argument(
            "--dataset",
            type=str,
            default="ogbn-arxiv",
            required=False,
            help="The input dataset.",
            choices=[
                "Flickr",
                "Reddit",
                "ogbn-products",
                "ogbn-papers100M",
                "Yelp",
                "AmazonProducts",
                "ogbn-arxiv",
                "cora",
                "citeseer",
                "pubmed"
            ],
        )

        parser.add_argument(
            "--type_model",
            type=str,
            default='EnGCN',
            choices=[
                "GraphSAGE",
                "FastGCN",
                "LADIES",
                "ClusterGCN",
                "GraphSAINT",
                "SGC",
                "SIGN",
                "SIGN_MLP",
                "LP_Adj",
                "SAGN",
                "GAMLP",
                "EnGCN",
            ],
        )
        parser.add_argument("--exp_name", type=str, default="")
        parser.add_argument("--N_exp", type=int, default=3)
        parser.add_argument("--resume", action="store_true", default=False)
        parser.add_argument(
            "--cuda", type=bool, default=True, required=False, help="run in cuda mode"
        )
        parser.add_argument("--cuda_num", type=int,
                            default=0, help="GPU number")
        parser.add_argument("--LM_emb_path", type=str, default=None,
                            help="Whether to load from the LM model")
        parser.add_argument("--GIANT", type=str, default=None,
                            help="GIANT-Feature to use")
        parser.add_argument("--seed", default=None, type=int)
        parser.add_argument("--num_layers", type=int, default=8)
        parser.add_argument(
            "--epochs",
            type=int,
            default=50,
            help="number of training the one shot model",
        )
        parser.add_argument(
            "--eval_steps",
            type=int,
            default=5,
            help="interval steps to evaluate model performance",
        )

        parser.add_argument(
            "--multi_label",
            type=bool,
            default=False,
            help="multi_label or single_label task",
        )
        parser.add_argument(
            "--dropout", type=float, default=0.2, help="input feature dropout"
        )
        parser.add_argument("--norm", type=str, default="None")
        parser.add_argument("--lr", type=float,
                            default=0.001, help="learning rate")
        parser.add_argument(
            "--weight_decay", type=float, default=0.0, help="weight decay"
        )  # 5e-4
        parser.add_argument("--dim_hidden", type=int, default=128)
        parser.add_argument(
            "--batch_size",
            type=int,
            default=5000,
            help="batch size depending on methods, "
                 "need to provide fair batch for different approaches",
        )
        # parameters for GraphSAINT
        parser.add_argument(
            "--walk_length", type=int, default=2, help="walk length of RW sampler"
        )
        parser.add_argument("--num_steps", type=int, default=5)
        parser.add_argument("--sample_coverage", type=int, default=0)
        parser.add_argument("--use_norm", type=bool, default=False)
        # parameters for ClusterGCN
        parser.add_argument("--num_parts", type=int, default=1500)
        # parameters for Greedy Gradient Sampling Selection
        parser.add_argument(
            "--dst_sample_coverage", type=float, default=0.1, help="dst sampling rate"
        )
        parser.add_argument(
            "--dst_walk_length", type=int, default=2, help="random walk length"
        )
        parser.add_argument(
            "--dst_update_rate",
            type=float,
            default=0.8,
            help="initialized dst update rate",
        )
        parser.add_argument(
            "--dst_update_interval", type=int, default=1, help="dst update interval"
        )
        parser.add_argument("--dst_T_end", type=int, default=250)
        parser.add_argument(
            "--dst_update_decay",
            type=bool,
            default=True,
            help="whether to decay update rate",
        )
        parser.add_argument(
            "--dst_update_scheme", type=str, default="node3", help="update schemes"
        )
        parser.add_argument(
            "--dst_grads_scheme",
            type=int,
            default=3,
            help="tem: search for updating scheme with grads",
        )

        parser.add_argument("--LP__no_prep", type=int,
                            default=0)  # no change!!!
        parser.add_argument(
            "--LP__pre_num_propagations", type=int, default=10
        )  # no change!!!
        parser.add_argument("--LP__A1", type=str,
                            default="DA")  # ['DA' 'AD' 'DAD']
        parser.add_argument("--LP__A2", type=str,
                            default="AD")  # ['DA' 'AD' 'DAD']
        parser.add_argument("--LP__prop_fn", type=int, default=1)  # [0,1]
        parser.add_argument("--LP__num_propagations1", type=int, default=50)
        parser.add_argument("--LP__num_propagations2", type=int, default=50)
        parser.add_argument("--LP__alpha1", type=float,
                            default=0.9791632871592579)
        parser.add_argument("--LP__alpha2", type=float,
                            default=0.7564990804200602)
        parser.add_argument("--LP__num_layers", type=int,
                            default=3)  # [0,  1,2,3]

        parser.add_argument("--SLE_threshold", type=float, default=0.9)

        parser.add_argument("--num_mlp_layers", type=int, default=3)
        parser.add_argument("--use_batch_norm", type=bool, default=True)
        parser.add_argument("--num_heads", type=int, default=1)
        parser.add_argument("--use_label_mlp", type=bool, default=True)

        parser.add_argument("--GAMLP_type", type=str,
                            default="JK", choices=["JK", "R"])
        parser.add_argument("--GAMLP_alpha", type=float, default=0.5)
        parser.add_argument("--GPR_alpha", type=float, default=0.1)
        parser.add_argument(
            "--GPR_init",
            type=str,
            default="PPR",
            choices=["SGC", "PPR", "NPPR", "Random", "WS", "Null"],
        )  # [0,  1,2,3]
        # hyperparameters for gradient evaluation
        parser.add_argument(
            "--type_run", type=str, default="filtered", choices=["complete", "filtered"]
        )
        parser.add_argument("--filter_rate", type=float, default=0.2)

        parser.add_argument("--alpha", type=float, default=1.0)

        args = parser.parse_args()
        args = self.reset_dataset_dependent_parameters(args)

        return args

    # setting the common hyperparameters used for comparing different methods of a trick
    def reset_dataset_dependent_parameters(self, args):
        if args.dataset == "ogbn-products":
            args.multi_label = False
            args.num_classes = 47
            if args.LM_emb_path or args.GIANT is not None:
                args.num_feats = 768
            else:
                args.num_feats = 100
        elif args.dataset == "ogbn-arxiv":
            args.num_classes = 40
            args.N_nodes = 169343
            if args.LM_emb_path or args.GIANT is not None:
                args.num_feats = 768
            else:
                args.num_feats = 128
        elif args.dataset == 'cora':
            args.num_classes = 8
            args.num_feats = 1433
            # args.num_feats = 128
        elif args.dataset == 'citeseer':
            args.num_classes = 6
            args.num_feats = 3703
        elif args.dataset == 'pubmed':
            args.num_classes = 3
            args.num_feats = 500
        return args
