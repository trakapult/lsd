import torch
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from data import CLEVRTEXDataset
from autoencoder import AutoencoderKL
from slotencoder import SlotEncoder
from unet import UNet
from lsd import LSD

parser = argparse.ArgumentParser()

parser.add_argument("--device", type=str, default="cuda:0")

parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--data_path", type=str, default="0")
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--grad_clip", type=float, default=1.0)
parser.add_argument("--save_step", type=int, default=500)
parser.add_argument("--valid_step", type=int, default=500)

parser.add_argument("--ae_embed_dim", type=int, default=4)
parser.add_argument("--ae_ckpt_path", type=str, default="model_ae_2.ckpt")

parser.add_argument("--se_num_iterations", type=int, default=3)
parser.add_argument("--se_num_slots", type=int, default=11)
parser.add_argument("--se_cnn_hidden_size", type=int, default=64)
parser.add_argument("--se_slot_size", type=int, default=128)
parser.add_argument("--se_mlp_hidden_size", type=int, default=128)
parser.add_argument("--se_img_channels", type=int, default=3)
parser.add_argument("--se_image_size", type=int, default=256)
parser.add_argument("--se_d_model", type=int, default=128)

parser.add_argument("--dif_T", type=int, default=1000)
parser.add_argument("--dif_ch", type=int, default=4)
parser.add_argument("--dif_image_size", type=list, default=32)
parser.add_argument("--dif_ch_mult", type=list, default=[1, 2, 4])
parser.add_argument("--dif_attn", type=list, default=[1])
parser.add_argument("--dif_num_res_blocks", type=int, default=2)
parser.add_argument("--dif_dropout", type=float, default=0.1)
parser.add_argument("--dif_scaling", type=float, default=0.018215)

parser.add_argument("--dif_beta_1", type=float, default=1e-4)
parser.add_argument("--dif_beta_T", type=float, default=0.02)
args = parser.parse_args()

torch.manual_seed(args.seed)

ddconfig = {"double_z": True,
            "z_channels": 4,
            "resolution": args.se_image_size,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 128,
            "ch_mult": [1, 2, 4, 4],
            "num_res_blocks": 2,
            "attn_resolutions": [],
            "dropout": 0.0}
tfm = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.Resize((args.se_image_size, args.se_image_size)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1)
])

dataset = CLEVRTEXDataset(args.data_path, tfm)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

model_ae = AutoencoderKL(ddconfig=ddconfig,
                         embed_dim=args.ae_embed_dim,
                         ckpt_path=args.ae_ckpt_path)
for param in model_ae.parameters():
    param.requires_grad = False
model_se = SlotEncoder(args)
model_dif = UNet(args)
model = LSD(model_ae, model_se, model_dif, args)
model = model.to(args.device)
optim = torch.optim.Adam(model.parameters(), lr=args.lr)

step = 0
for epoch in range(args.num_epochs):
    for image in dataloader:
        step += 1
        image = image.to(args.device)
        model.train()
        optim.zero_grad()
        loss = model(image)
        print("Epoch: {}, Step: {}, Loss: {}".format(epoch, step, loss))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()
        if (step + 1) % args.save_step == 0:
            torch.save(model.state_dict(), "model.ckpt")
        if (step + 1) % args.valid_step == 0:
            model.eval()
            with torch.no_grad():
                pred = model.recon(image)
                print(f"step{step}.png")
                grid = make_grid((pred + 1) / 2, nrow=4)
                print((pred[0] + 1) / 2)
                save_image(grid, f"step{step}.png")