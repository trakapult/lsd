import torch
import torch.nn as nn
from utils import *

class SlotAttention(nn.Module):
    
    def __init__(self, num_iterations, num_slots,
                 input_size, slot_size, mlp_hidden_size,
                 epsilon=1e-8):
        super().__init__()
        
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.input_size = input_size
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon

        # parameters for Gaussian initialization (shared by all slots).
        self.slot_mu = nn.Parameter(torch.Tensor(1, 1, slot_size))
        self.slot_log_sigma = nn.Parameter(torch.Tensor(1, 1, slot_size))
        nn.init.xavier_uniform_(self.slot_mu)
        nn.init.xavier_uniform_(self.slot_log_sigma)

        # norms
        self.norm_inputs = nn.LayerNorm(input_size)
        self.norm_slots = nn.LayerNorm(slot_size)
        self.norm_mlp = nn.LayerNorm(slot_size)
        
        # linear maps for the attention module.
        self.project_q = linear(slot_size, slot_size, bias=False)
        self.project_k = linear(input_size, slot_size, bias=False)
        self.project_v = linear(input_size, slot_size, bias=False)
        
        # slot update functions.
        self.gru = gru_cell(slot_size, slot_size)
        self.mlp = nn.Sequential(
            linear(slot_size, mlp_hidden_size, weight_init='kaiming'),
            nn.ReLU(),
            linear(mlp_hidden_size, slot_size))

    def forward(self, inputs):
        B = inputs.shape[0]

        # initialize slots
        slots = inputs.new_empty(B, self.num_slots, self.slot_size).normal_()
        slots = self.slot_mu + torch.exp(self.slot_log_sigma) * slots

        # setup key and value
        #input要先变成特征图的，STEVE是用CNN，LSD是用了一个Unet
        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs)  # Shape: [batch_size, T, num_inputs, slot_size].
        v = self.project_v(inputs)  # Shape: [batch_size, T, num_inputs, slot_size].
        k = (self.slot_size ** (-0.5)) * k
        
        for i in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention.
            q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
            attn_logits = torch.bmm(k, q.transpose(-1, -2))
            attn_vis = F.softmax(attn_logits, dim=-1)
            # `attn_vis` has shape: [batch_size, num_inputs, num_slots].

            # Weighted mean.
            attn = attn_vis + self.epsilon
            attn = attn / torch.sum(attn, dim=-2, keepdim=True)
            updates = torch.bmm(attn.transpose(-1, -2), v)
            # `updates` has shape: [batch_size, num_slots, slot_size].

            # Slot update.
            slots = self.gru(updates.view(-1, self.slot_size),
                             slots_prev.view(-1, self.slot_size))
            slots = slots.view(-1, self.num_slots, self.slot_size)

            # use MLP only when more than one iterations
            if i < self.num_iterations - 1:
                slots = slots + self.mlp(self.norm_mlp(slots))
		
        return slots

class CartesianPositionalEmbedding(nn.Module):

    def __init__(self, channels, image_size):
        super().__init__()

        self.projection = conv2d(4, channels, 1)
        self.pe = nn.Parameter(self.build_grid(image_size).unsqueeze(0), requires_grad=False)

    def build_grid(self, side_length):
        coords = torch.linspace(0., 1., side_length + 1)
        coords = 0.5 * (coords[:-1] + coords[1:])
        grid_y, grid_x = torch.meshgrid(coords, coords)
        return torch.stack((grid_x, grid_y, 1 - grid_x, 1 - grid_y), dim=0)

    def forward(self, inputs):
        # `inputs` has shape: [batch_size, out_channels, height, width].
        # `grid` has shape: [batch_size, in_channels, height, width].
        return inputs + self.projection(self.pe)

class SlotEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.num_iterations = args.se_num_iterations
        self.num_slots = args.se_num_slots
        self.cnn_hidden_size = args.se_cnn_hidden_size
        self.slot_size = args.se_slot_size
        self.mlp_hidden_size = args.se_mlp_hidden_size
        self.img_channels = args.se_img_channels
        self.image_size = args.se_image_size
        self.d_model = args.se_d_model
        self.cnn = nn.Sequential(
            Conv2dBlock(self.img_channels, self.cnn_hidden_size, 5, 1 if self.image_size == 64 else 2, 2),
            Conv2dBlock(self.cnn_hidden_size, self.cnn_hidden_size, 5, 1, 2),
            Conv2dBlock(self.cnn_hidden_size, self.cnn_hidden_size, 5, 1, 2),
            conv2d(self.cnn_hidden_size, self.d_model, 5, 1, 2),
        )

        self.pos = CartesianPositionalEmbedding(self.d_model, self.image_size if self.image_size == 64 else self.image_size // 2)

        self.layer_norm = nn.LayerNorm(self.d_model)

        self.mlp = nn.Sequential(
            linear(self.d_model, self.d_model, weight_init='kaiming'),
            nn.ReLU(),
            linear(self.d_model, self.d_model))

        self.sa = SlotAttention(
            self.num_iterations, self.num_slots,
            self.d_model, self.slot_size, self.mlp_hidden_size)

        self.slot_proj = linear(self.slot_size, self.d_model, bias=False)

    def forward(self, image):
        B, C, H, W = image.size()
        
        emb = self.cnn(image)      # B * T, cnn_hidden_size, H, W
        emb = self.pos(emb)             # B * T, cnn_hidden_size, H, W

        emb_set = emb.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)                                   # B * T, H * W, cnn_hidden_size
        emb_set = self.mlp(self.layer_norm(emb_set))                            # B * T, H * W, cnn_hidden_size

        slots = self.sa(emb_set)     # slots: B, T, num_slots, slot_size

        return slots