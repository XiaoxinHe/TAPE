import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BatchNorm1d, Linear
from torch.utils.data import DataLoader


class MLP_SLE(torch.nn.Module):
    def __init__(self, args) -> None:
        super(MLP_SLE, self).__init__()
        self.use_label_mlp = args.use_label_mlp
        self.base_mlp = Inner_MLP(
            args.num_feats,
            args.dim_hidden,
            args.num_classes,
            args.num_mlp_layers,
            args.dropout,
            args.use_batch_norm,
        )
        if self.use_label_mlp:
            self.label_mlp = GroupMLP(
                args.num_classes,
                args.dim_hidden,
                args.num_classes,
                args.num_heads,
                args.num_mlp_layers,
                args.dropout,
                normalization="batch" if args.use_batch_norm else "none",
            )
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        self.loss_op = torch.nn.NLLLoss()
        self.alpha = args.alpha

    def forward(self, x, y, use_label_mlp):
        out = self.base_mlp(x)
        if use_label_mlp:
            out += self.label_mlp(y).mean(1)
        return out

    def load(self):
        self.base_mlp.load_state_dict(
            torch.load(
                f"./.cache/{self.type_model}_{self.dataset}_base_mlp_LM.pt")
        )
        self.label_mlp.load_state_dict(
            torch.load(
                f"./.cache/{self.type_model}_{self.dataset}_label_mlp_LM.pt")
        )

    def train_net(self, train_loader, loss_op, device, use_label_mlp, model_z, optimizer_z, split_mask, batch_size):
        self.train()
        model_z.train()
        cos_sim = torch.nn.CosineSimilarity()
        total_correct, total_loss, total_loss0, total_loss1 = 0, 0.0, 0.0, 0.0
        y_true, y_preds = [], []

        for batch_idx, (y_emb, y, lm_z) in enumerate(train_loader):
            z = model_z()[split_mask["train"]][batch_idx *
                                               batch_size: (batch_idx+1)*batch_size]
            z.to(device)
            y = y.to(device)
            y_emb = y_emb.to(device)
            lm_z = lm_z.to(device)
            self.optimizer.zero_grad()
            optimizer_z.zero_grad()
            out = self(z, y_emb, use_label_mlp)
            if isinstance(loss_op, torch.nn.NLLLoss):
                out = F.log_softmax(out, dim=-1)
            elif isinstance(loss_op, torch.nn.BCEWithLogitsLoss):
                y = y.float()
            loss0 = loss_op(out, y)
            loss0 = loss0.mean()
            loss1 = self.alpha * (1 - cos_sim(z, lm_z).mean())
            loss = loss0 + loss1
            loss.backward()
            total_loss += float(loss.item())
            total_loss0 += float(loss0.item())
            total_loss1 += float(loss1.item())
            self.optimizer.step()
            optimizer_z.step()
            y_preds.append(out.argmax(dim=-1).detach().cpu())
            y_true.append(y.detach().cpu())

        y_true = torch.cat(y_true, 0)
        y_preds = torch.cat(y_preds, 0)
        total_correct = y_preds.eq(y_true).sum().item()
        train_acc = float(total_correct / y_preds.size(0))
        return float(total_loss), float(total_loss0), float(total_loss1), train_acc

    @torch.no_grad()
    def inference(self, x, y_emb, device, use_label_mlp):
        self.eval()
        loader = DataLoader(range(x.size(0)), batch_size=100000)
        outs = []
        for perm in loader:
            out = self(x[perm].to(device),
                       y_emb[perm].to(device), use_label_mlp)
            outs.append(out.cpu())
        return torch.cat(outs, dim=0)


class Inner_MLP(torch.nn.Module):
    def __init__(
        self, in_dim, hidden_dim, out_dim, num_layers, dropout, use_batch_norm
    ):
        super(Inner_MLP, self).__init__()
        self.linear_list = torch.nn.ModuleList()
        self.batch_norm_list = torch.nn.ModuleList()
        self.dropout = dropout
        self.num_layers = num_layers
        self.use_batch_norm = use_batch_norm

        self.linear_list.append(Linear(in_dim, hidden_dim))
        self.batch_norm_list.append(BatchNorm1d(hidden_dim))
        for _ in range(self.num_layers - 2):
            self.linear_list.append(Linear(hidden_dim, hidden_dim))
            self.batch_norm_list.append(BatchNorm1d(hidden_dim))
        self.linear_list.append(Linear(hidden_dim, out_dim))

    def forward(self, x):
        for i in range(self.num_layers - 1):
            x = self.linear_list[i](x)
            if self.use_batch_norm:
                x = self.batch_norm_list[i](x)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        return self.linear_list[-1](x)


# classes from SAGN
class MultiHeadLinear(nn.Module):
    def __init__(self, in_feats, out_feats, n_heads, bias=True):
        super().__init__()
        self.weight = nn.Parameter(
            torch.FloatTensor(size=(n_heads, in_feats, out_feats))
        )
        if bias:
            self.bias = nn.Parameter(
                torch.FloatTensor(size=(n_heads, 1, out_feats)))
        else:
            self.bias = None

    def reset_parameters(self) -> None:
        for weight, bias in zip(self.weight, self.bias):
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            if bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(bias, -bound, bound)

    # def reset_parameters(self):
    #     gain = nn.init.calculate_gain("relu")
    #     for weight in self.weight:
    #         nn.init.xavier_uniform_(weight, gain=gain)
    #     if self.bias is not None:
    #         nn.init.zeros_(self.bias)

    def forward(self, x):
        # input size: [N, d_in] or [H, N, d_in]
        # output size: [H, N, d_out]
        if len(x.shape) == 3:
            x = x.transpose(0, 1)

        x = torch.matmul(x, self.weight)
        if self.bias is not None:
            x += self.bias
        return x.transpose(0, 1)


# Modified multi-head BatchNorm1d layer
class MultiHeadBatchNorm(nn.Module):
    def __init__(
        self, n_heads, in_feats, momentum=0.1, affine=True, device=None, dtype=None
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        assert in_feats % n_heads == 0
        self._in_feats = in_feats
        self._n_heads = n_heads
        self._momentum = momentum
        self._affine = affine
        if affine:
            self.weight = nn.Parameter(torch.empty(
                size=(n_heads, in_feats // n_heads)))
            self.bias = nn.Parameter(torch.empty(
                size=(n_heads, in_feats // n_heads)))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.register_buffer(
            "running_mean", torch.zeros(size=(n_heads, in_feats // n_heads))
        )
        self.register_buffer(
            "running_var", torch.ones(size=(n_heads, in_feats // n_heads))
        )
        self.running_mean: Optional[Tensor]
        self.running_var: Optional[Tensor]
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()  # type: ignore[union-attr]
        self.running_var.fill_(1)  # type: ignore[union-attr]
        if self._affine:
            nn.init.zeros_(self.bias)
            for weight in self.weight:
                nn.init.ones_(weight)

    def forward(self, x, eps=1e-5):
        assert x.shape[1] == self._in_feats
        x = x.view(-1, self._n_heads, self._in_feats // self._n_heads)

        self.running_mean = self.running_mean.to(x.device)
        self.running_var = self.running_var.to(x.device)
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (
                self.running_var is None)
        if bn_training:
            mean = x.mean(dim=0, keepdim=True)
            var = x.var(dim=0, unbiased=False, keepdim=True)
            out = (x - mean) * torch.rsqrt(var + eps)
            self.running_mean = (
                1 - self._momentum
            ) * self.running_mean + self._momentum * mean.detach()
            self.running_var = (
                1 - self._momentum
            ) * self.running_var + self._momentum * var.detach()
        else:
            out = (x - self.running_mean) * torch.rsqrt(self.running_var + eps)
        if self._affine:
            out = out * self.weight + self.bias
        return out


class GroupMLP(nn.Module):
    def __init__(
        self,
        in_feats,
        hidden,
        out_feats,
        n_heads,
        n_layers,
        dropout,
        input_drop=0.0,
        residual=False,
        normalization="batch",
    ):
        super(GroupMLP, self).__init__()
        self._residual = residual
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self._n_heads = n_heads
        self._n_layers = n_layers

        self.input_drop = nn.Dropout(input_drop)

        if self._n_layers == 1:
            self.layers.append(MultiHeadLinear(in_feats, out_feats, n_heads))
        else:
            self.layers.append(MultiHeadLinear(in_feats, hidden, n_heads))
            if normalization == "batch":
                self.norms.append(MultiHeadBatchNorm(
                    n_heads, hidden * n_heads))
                # self.norms.append(nn.BatchNorm1d(hidden * n_heads))
            if normalization == "layer":
                self.norms.append(nn.GroupNorm(n_heads, hidden * n_heads))
            if normalization == "none":
                self.norms.append(nn.Identity())
            for i in range(self._n_layers - 2):
                self.layers.append(MultiHeadLinear(hidden, hidden, n_heads))
                if normalization == "batch":
                    self.norms.append(MultiHeadBatchNorm(
                        n_heads, hidden * n_heads))
                    # self.norms.append(nn.BatchNorm1d(hidden * n_heads))
                if normalization == "layer":
                    self.norms.append(nn.GroupNorm(n_heads, hidden * n_heads))
                if normalization == "none":
                    self.norms.append(nn.Identity())
            self.layers.append(MultiHeadLinear(hidden, out_feats, n_heads))
        if self._n_layers > 1:
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

        for head in range(self._n_heads):

            for layer in self.layers:

                nn.init.kaiming_uniform_(layer.weight[head], a=math.sqrt(5))
                if layer.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                        layer.weight[head]
                    )
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(layer.bias[head], -bound, bound)
        self.reset_parameters()

    def reset_parameters(self):

        gain = nn.init.calculate_gain("relu")

        for head in range(self._n_heads):
            for layer in self.layers:
                nn.init.xavier_uniform_(layer.weight[head], gain=gain)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias[head])
        for norm in self.norms:
            norm.reset_parameters()
            # for norm in self.norms:
            #     norm.moving_mean[head].zero_()
            #     norm.moving_var[head].fill_(1)
            #     if norm._affine:
            #         nn.init.ones_(norm.scale[head])
            #         nn.init.zeros_(norm.offset[head])
        # print(self.layers[0].weight[0])

    def forward(self, x):
        x = self.input_drop(x)
        if len(x.shape) == 2:
            x = x.view(-1, 1, x.shape[1])
        if self._residual:
            prev_x = x
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)

            if layer_id < self._n_layers - 1:
                shape = x.shape
                x = x.flatten(1, -1)
                x = self.dropout(self.relu(self.norms[layer_id](x)))
                x = x.reshape(shape=shape)

            if self._residual:
                if x.shape[2] == prev_x.shape[2]:
                    x += prev_x
                prev_x = x

        return x
