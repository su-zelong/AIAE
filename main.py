import torch
import logging
import torch.nn.functional as F
import torch.optim as optim

from arguments import Argument
from load_data import load_data
from model.AIAE import Encoder, Decoder_imputation, Decoder_MS
from model.model_utils import create_adj

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def imputation(z, h):
    return torch.concat((h, z), dim=0)


def scaled_cosine_error(y_true, y_pred):
    norm_y_true = torch.norm(y_true, p=2, dim=1, keepdim=True)
    norm_y_pred = torch.norm(y_pred, p=2, dim=1, keepdim=True)    
    dot_product = (y_true * y_pred).sum(dim=1, keepdim=True)    
    sce = 1 - dot_product / (norm_y_true * norm_y_pred)    
    sce = torch.clamp(sce, min=0)
    return sce.mean()

def figure_loss_lc(args, x__, z_2, train_x, train_mask):
    l_mse = torch.sum((x__[train_mask] - z_2[train_mask]) ** 2) / len(train_x)
    l_sce = scaled_cosine_error(z_2[train_mask], x__[train_mask])
    return l_mse + l_sce


def figure_loss_lz(args, z_2, x_, train_mask, train_x, adj_, edges):
    loss_x = torch.sum((z_2[train_mask] - x_[train_mask]) ** 2) / len(train_x)
    adj = create_adj(edges)
    loss_adj = args.l_z * F.binary_cross_entropy(adj, adj_) / (len(adj) ** 2)
    return loss_x + loss_adj


def figure_loss_ld(args, h1_0, h2_0, z_1, z_2, train_mask):
    z1_0, z2_0 = z_1[train_mask], z_2[train_mask]
    d1_1, d1_2 = F.softmax(z1_0 / args.l_d), F.softmax(h1_0 / args.l_d)
    l_d1 = F.kl_div(d1_1, d1_2, reduction='batchmean')
    d2_1, d2_2 = F.softmax(z2_0 / args.l_d), F.softmax(h2_0 / args.l_d)
    l_d2 = F.kl_div(d2_1, d2_2, reduction='batchmean')
    l_d = l_d1 + l_d2
    return l_d

def main():
    args = Argument().get_args()
    torch.manual_seed(args.seed)
    train_x, train_y, train_mask, val_x, val_y, val_mask, test_x, test_y, test_mask, train_edges, edges = load_data(args)
    args.train_length, args.val_length, args.test_length = int(train_x.shape[0]), int(val_x.shape[0]), int(test_x.shape[0])
    args.node_dim = train_x.shape[1]
    args.node_length = args.train_length + args.val_length + args.test_length
    args.missing_length = args.val_length + args.test_length

    # profilling model
    encoder_model = Encoder(args)
    imp_decoder = Decoder_imputation(args)
    ms_decoder = Decoder_MS(args)

    encoder_params = list(encoder_model.parameters())
    imp_params = list(imp_decoder.parameters())
    ms_params = list(ms_decoder.parameters())
    optimizer = optim.Adam(encoder_params + imp_params + ms_params, lr=args.lr, weight_decay=args.weight_decay)

    for i in range(args.epoch):
        encoder_model.train()
        imp_decoder.train()

        optimizer.zero_grad()
        # encoder
        h1_0, h2_0, z_1, z_2 = encoder_model(train_x, train_edges, train_mask, edges)
        l_d = figure_loss_ld(args, h1_0, h2_0, z_1, z_2, train_mask)
        z_1_, z_2_ = imputation(h1_0, z_1[~train_mask]), imputation(h2_0, z_2[~train_mask])
        
        # decoder-imputation: 需要将维度转换为 3703
        x_, adj_ = imp_decoder(z_2, z_2_)    # 重新生成词向量 x_.shape=1620*64 和 adj
        l_z = figure_loss_lz(args, z_2, x_, train_mask, train_x, adj_, edges)    # 不能直接用one-hot生成

        # decoder multi scale
        x__ = ms_decoder(z_1_, z_2_, train_mask, edges)
        l_c = figure_loss_lc(args, x__, z_2, train_x, train_mask)

        loss = args.a * l_d + args.b * l_z + args.c * l_c
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    main()
