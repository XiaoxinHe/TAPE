import gc
import os
import torch
import torch.nn.functional as F
from core.model_utils.WeakLearners_LM import MLP_SLE
from core.model import Z
import time


class EnGCN(torch.nn.Module):
    def __init__(self, args, data, evaluator, init_z):
        super(EnGCN, self).__init__()
        # first try multiple weak learners
        self.model = MLP_SLE(args)
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f'EnGCN trainable_params are {trainable_params}')

        self.evaluator = evaluator
        self.SLE_threshold = args.SLE_threshold
        self.use_label_mlp = args.use_label_mlp
        self.type_model = args.type_model
        self.dataset = args.dataset
        self.num_layers = args.num_layers
        self.num_classes = args.num_classes
        self.num_feats = args.num_feats
        self.batch_size = args.batch_size
        self.dim_hidden = args.dim_hidden
        self.dropout = args.dropout
        self.epochs = args.epochs
        self.multi_label = args.multi_label
        self.interval = args.eval_steps
        self.exp_name = args.exp_name
        # new
        self.model_z = Z(init_z)

        deg = data.adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        self.adj_t = deg_inv_sqrt.view(-1, 1) * \
            data.adj_t * deg_inv_sqrt.view(1, -1)

        del data, deg, deg_inv_sqrt
        gc.collect()

    def forward(self, x):
        pass

    def propagate(self, x):
        if x.dtype == torch.bfloat16:
            x = x.to(torch.float)
            x = self.adj_t @ x
            return x.to(torch.bfloat16)
        else:
            return self.adj_t @ x

    def to(self, device):
        self.model.to(device)

    def train_and_test(self, input_dict):

        device, split_masks, x, y, loss_op, lm_z = (
            input_dict["device"],
            input_dict["split_masks"],
            input_dict["x"],
            input_dict["y"],
            input_dict["loss_op"],
            input_dict["lm_z"]
        )

        del input_dict
        gc.collect()
        self.to(device)

        print(f"dtype y: {y.dtype}")
        results = torch.zeros(y.size(0), self.num_classes)
        y_emb = torch.zeros(y.size(0), self.num_classes)
        y_emb[split_masks["train"]] = F.one_hot(
            y[split_masks["train"]], num_classes=self.num_classes
        ).to(torch.float)
        # for self training
        pseudo_labels = torch.zeros_like(y)
        pseudo_labels[split_masks["train"]] = y[split_masks["train"]]
        pseudo_split_masks = split_masks

        print(
            "------ pseudo labels inited, rate: {:.4f} ------".format(
                pseudo_split_masks["train"].sum() / len(y)
            )
        )

        for i in range(self.num_layers):
            # NOTE: here the num_layers should be the stages in original SAGN
            with torch.no_grad():
                self.model_z.Z = torch.nn.Parameter(x.clone())
                self.optimizer_z = torch.optim.Adam(
                    self.model_z.parameters(), lr=1e-4)
            print(f"\n------ training weak learner with hop {i} ------")
            self.train_weak_learner(
                i,
                y_emb,
                pseudo_labels,
                y,  # the ground truth
                # ['train'] is pseudo, valide and test are not modified
                pseudo_split_masks,
                device,
                loss_op,
                lm_z
            )
            self.model.load_state_dict(
                torch.load(
                    f"./.cache/{self.exp_name}_{self.dataset}_MLP_SLE_TAG.pt")
            )
            self.model_z.load_state_dict(
                torch.load(
                    f"./.cache/{self.exp_name}_{self.dataset}_Z_TAG.pt")
            )
            # make prediction
            use_label_mlp = False if i == 0 else self.use_label_mlp

            self.model_z.eval()
            z = self.model_z()
            out = self.model.inference(z, y_emb, device, use_label_mlp)

            # self training: add hard labels
            val, pred = torch.max(F.softmax(out, dim=1), dim=1)
            SLE_mask = val >= self.SLE_threshold
            SLE_pred = pred[SLE_mask]

            # SLE_pred U y
            pseudo_split_masks["train"] = pseudo_split_masks["train"].logical_or(
                SLE_mask
            )
            pseudo_labels[SLE_mask] = SLE_pred
            pseudo_labels[split_masks["train"]] = y[split_masks["train"]]

            # update y_emb
            # y_emb[pseudo_split_masks["train"]] = F.one_hot(
            #     pseudo_labels[pseudo_split_masks["train"]], num_classes=self.num_classes
            # ).to(torch.float)

            del val, pred, SLE_mask, SLE_pred
            gc.collect()
            # y_emb = self.propagate(y_emb.to(device))
            # x = self.propagate(x.to(device))
            y_emb = self.propagate(y_emb)
            x = self.propagate(x)

            print(
                "------ pseudo labels updated, rate: {:.4f} ------".format(
                    pseudo_split_masks["train"].sum() / len(y)
                )
            )

            # NOTE: adaboosting (SAMME.R)
            out = F.log_softmax(out, dim=1)
            results += (self.num_classes - 1) * (
                out - torch.mean(out, dim=1).view(-1, 1)
            )
            del out

        out, acc = self.evaluate(results, y, split_masks)
        out = F.softmax(out, dim=1)
        print(
            f"Final train acc: {acc['train']*100:.4f}, "
            f"Final valid acc: {acc['valid']*100:.4f}, "
            f"Dianl test acc: {acc['test']*100:.4f}"
        )
        # dirs = f"./output/{self.dataset}/"
        # if not os.path.exists(dirs):
        #     os.makedirs(dirs)
        # checkpt_file = dirs + uuid.uuid4().hex
        # torch.save(out, checkpt_file + f'EnGCN.pt')
        return acc["train"], acc["valid"], acc["test"]

    def evaluate(self, out, y, split_mask):
        acc = {}
        if self.evaluator:
            y_true = y.unsqueeze(-1)
            y_pred = out.argmax(dim=-1, keepdim=True)
            for phase in ["train", "valid", "test"]:
                acc[phase] = self.evaluator.eval(
                    {
                        "y_true": y_true[split_mask[phase]],
                        "y_pred": y_pred[split_mask[phase]],
                    }
                )["acc"]
        else:
            pred = out.argmax(dim=1).to("cpu")
            y_true = y
            correct = pred.eq(y_true)
            for phase in ["train", "valid", "test"]:
                acc[phase] = (
                    correct[split_mask[phase]].sum().item()
                    / split_mask[phase].sum().item()
                )
        return out, acc

    def train_weak_learner(self, hop, y_emb, pseudo_labels, origin_labels, split_mask, device, loss_op, lm_z):
        pesudo_labels_train = pseudo_labels[split_mask["train"]]
        y_emb_train = y_emb[split_mask["train"]]
        lm_z_train = lm_z[split_mask["train"]]

        best_valid_acc = 0.0
        use_label_mlp = self.use_label_mlp
        if hop == 0:
            use_label_mlp = False  # warm up
        for epoch in range(self.epochs):
            start = time.time()
            loss, loss0, loss1, _train_acc = self.model.train_net(
                y_emb_train, pesudo_labels_train, lm_z_train, loss_op, device, use_label_mlp, self.model_z, self.optimizer_z, split_mask, self.batch_size)
            end = time.time()
            if (epoch + 1) % self.interval == 0:
                use_label_mlp = False if hop == 0 else self.use_label_mlp
                x = self.model_z()
                out = self.model.inference(x, y_emb, device, use_label_mlp)
                out, acc = self.evaluate(out, origin_labels, split_mask)
                print(
                    f"Model: {hop:02d}, "
                    f"Epoch: {epoch:02d}, "
                    f"Loss: {loss:.4f}, "
                    f"Loss(GNN): {loss0:.4f}, "
                    f"Loss(Z): {loss1:.8f}, "
                    f"Train acc: {acc['train']*100:.4f}, "
                    f"Valid acc: {acc['valid']*100:.4f}, "
                    f"Test acc: {acc['test']*100: .4f}, "
                    f"Time: {(end-start):.4f}"
                )
                if acc["valid"] > best_valid_acc:
                    best_valid_acc = acc["valid"]
                    if not os.path.exists(".cache/"):
                        os.mkdir(".cache/")
                    torch.save(
                        self.model.state_dict(),
                        f"./.cache/{self.exp_name}_{self.dataset}_MLP_SLE_TAG.pt",
                    )
                    torch.save(
                        self.model_z.state_dict(),
                        f"./.cache/{self.exp_name}_{self.dataset}_Z_TAG.pt",
                    )