import math
import random

from obj.env import Env
from models.gat import GAT

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# if gpu is to be used
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HouseCritic(nn.Module):

    def __init__(self):
        super(HouseCritic, self).__init__()

        self._emb_u_p = nn.Sequential(
            nn.Linear(in_features=Env.PROFILE_FEATURES_DIM, out_features=Env.PROFILE_EMBEDDING_DIM),
            nn.CELU(),
        )

        self._emb_u_s = nn.Sequential(
            nn.Linear(in_features=Env.SHOOPING_FEATURES_DIM, out_features=Env.SHOOPING_EMBEDDING_DIM),
            nn.CELU(),
        )

        self._emb_u_w = nn.Sequential(
            nn.Linear(in_features=Env.WORKSPACE_FEATURES_DIM, out_features=Env.WORKSPACE_EMBEDDING_DIM),
            nn.CELU(),
        )

        self.v = nn.Linear((Env.PROFILE_EMBEDDING_DIM+Env.SHOOPING_EMBEDDING_DIM)*2, 1, bias=False)

        self._fusion_u = nn.Sequential(
            nn.Linear(in_features=(2*Env.PROFILE_EMBEDDING_DIM+2*Env.SHOOPING_EMBEDDING_DIM+Env.WORKSPACE_EMBEDDING_DIM), out_features=Env.FUSION_HIDDEN),
            nn.CELU(),
            nn.Linear(in_features=Env.FUSION_HIDDEN, out_features=Env.FUSION_OUTPUT),
            nn.CELU(),
        )

        self._mlp_1 = nn.Sequential(
            nn.Linear(in_features=Env.FUSION_OUTPUT, out_features=(Env.PROPERTY_EMBEDDING_DIM+Env.VICINITY_EMBEDDING_DIM+Env.REACHABILITY_EMBEDDING_DIM+Env.PROFILE_EMBEDDING_DIM+Env.SHOOPING_EMBEDDING_DIM)*Env.META_HIDDEN_1),
            nn.CELU(),
        )

        self._mlp_2 = nn.Sequential(
            nn.Linear(in_features=Env.FUSION_OUTPUT, out_features=Env.META_HIDDEN_1*Env.META_HIDDEN_2),
            nn.CELU(),
        )

        self._mlp_3 = nn.Sequential(
            nn.Linear(in_features=Env.FUSION_OUTPUT, out_features=Env.META_HIDDEN_2),
            nn.CELU(),
        )

        self._emb_h_p = nn.Sequential(
            nn.Linear(in_features=Env.PROPERTY_FEATURES_DIM, out_features=Env.PROPERTY_EMBEDDING_DIM),
            nn.CELU(),
        )

        self._emb_h_v = nn.Sequential(
            nn.Linear(in_features=Env.VICINITY_FEATURES_DIM, out_features=Env.VICINITY_EMBEDDING_DIM),
            nn.CELU(),
        )

        self._emb_h_r = nn.Sequential(
            nn.Linear(in_features=Env.REACHABILITY_FEATURES_DIM, out_features=Env.REACHABILITY_EMBEDDING_DIM),
            nn.CELU(),
        )

        self.gat = GAT(nfeat=Env.PROFILE_EMBEDDING_DIM+Env.SHOOPING_EMBEDDING_DIM, nhid = Env.GAT_HIDDEN, nouput = Env.PROFILE_EMBEDDING_DIM+Env.SHOOPING_EMBEDDING_DIM, dropout = 0.6, nheads = 4, alpha = 0.2)

        self.activation = nn.ReLU()
        self.out_act = nn.Sigmoid()

    def meta_fcn(self, w, x):
        x = torch.mm(x, w[0])
        x = self.activation(x)
        x = torch.mm(x, w[1])
        x = self.activation(x)
        x = torch.mm(x, w[2])
        x = self.out_act(x)
        return x

    def forward(self, x_u_profile, x_u_shopping, x_u_workplace, x_u_colleague_p, x_u_colleague_s, x_h_property, x_h_vicinity, x_h_reachability, x_h_neighborhood_p, x_h_neighborhood_s, x_h_neighborhood_g):

        # user module
        ########embedding process
        x_u_profile = self._emb_u_p(x_u_profile)
        x_u_shopping = self._emb_u_s(x_u_shopping)
        x_u_workplace = self._emb_u_w(x_u_workplace)
        x_u_colleague_p = self._emb_u_p(x_u_colleague_p)
        x_u_colleague_s = self._emb_u_s(x_u_colleague_s)

        x_u_p_s_expended = torch.cat((x_u_shopping.expend(x_u_colleague_p.size(0), -1), x_u_profile.expend(x_u_colleague_s.size(0), -1)), dim=1)
        x_u_p_s_concat = torch.cat((x_u_p_s_expended, x_u_colleague_p, x_u_colleague_s), dim=1)
        atten = self.v(x_u_p_s_concat)
        atten = self.softmax(atten)
        x_colleague = torch.bmm(atten, torch.cat((x_u_colleague_p, x_u_colleague_s), dim=1))
        x_colleague = x_colleague.sum(1)/x_u_colleague_p.size(0)

        ########fusion process
        x_u = self._fusion_u(torch.cat((x_u_profile, x_u_shopping, x_u_workplace, x_colleague), dim=1))

        ########weight generation
        weight_1 = self._mlp_1(x_u).view((Env.PROPERTY_EMBEDDING_DIM+Env.VICINITY_EMBEDDING_DIM+Env.REACHABILITY_EMBEDDING_DIM+Env.PROFILE_EMBEDDING_DIM+Env.SHOOPING_EMBEDDING_DIM), Env.META_HIDDEN_1).contiguous()
        weight_2 = self._mlp_2(x_u).view(Env.META_HIDDEN_1, Env.META_HIDDEN_2).contiguous()
        weight_3 = self._mlp_2(x_u).view(Env.META_HIDDEN_2, 1).contiguous()


        # house module
        x_h_property = self._emb_h_p(x_h_property)
        x_h_vicinity= self._emb_h_v(x_h_vicinity)
        x_h_reachability = self._emb_h_r(x_h_reachability)
        x_h_neighborhood_p = self._emb_u_p(x_h_neighborhood_p)
        x_h_neighborhood_s = self._emb_u_s(x_h_neighborhood_s)
        x_h_neighborhood = self.gat(torch.cat((x_h_neighborhood_p, x_h_neighborhood_s), dim=1), x_h_neighborhood_g)
        x_h_neighborhood = x_h_neighborhood.view(1, -1)

        x_h = torch.cat((x_h_property, x_h_vicinity, x_h_reachability, x_h_neighborhood), dim=1)

        # selection module
        degree = self.meta_fcn([weight_1, weight_2, weight_3], x_h)

        return degree


def train_epoch(e_num, model, opt, criterion, x_u_profiles, x_u_shoppings, x_u_workplaces, x_u_colleague_ps, x_u_colleague_ss, x_h_properties, x_h_vicinities, x_h_reachabilities, x_h_neighborhood_ps, x_h_neighborhood_ss, x_h_neighborhood_gs, Y, batch_size = 1000):

    model.train()
    losses = []
    loss_batch = 0.0
    user_num = len(x_u_profiles)
    for i in range(0, user_num):
        
        x_u_profile, x_u_shopping, x_u_workplace, x_u_colleague_p, x_u_colleague_s, x_h_property, x_h_vicinity, x_h_reachability, x_h_neighborhood_p, x_h_neighborhood_s, x_h_neighborhood_g =  x_u_profiles[i], x_u_shoppings[i], x_u_workplaces[i], x_u_colleague_ps[i], x_u_colleague_ss[i], x_h_properties[i], x_h_vicinities[i], x_h_reachabilities[i], x_h_neighborhood_ps[i], x_h_neighborhood_ss[i], x_h_neighborhood_gs[i]
        y_batch = Y[i]

        
        # (1) Forward
        y_hat = model(x_u_profile, x_u_shopping, x_u_workplace, x_u_colleague_p, x_u_colleague_s, x_h_property, x_h_vicinity, x_h_reachability, x_h_neighborhood_p, x_h_neighborhood_s, x_h_neighborhood_g)
        # (2) Compute diff
        loss = criterion(y_hat, y_batch)
        # (3) Compute gradients
        losses.append(loss.data.numpy())
        loss_batch += loss
        # loss.backward()

        if i % batch_size == 0 and i!=0:
            opt.zero_grad()
            loss_batch /= batch_size
            loss_batch.backward()
            opt.step()
            print('[%d, %5d] loss: %.3f' %
                 (e_num + 1, i + 1, loss_batch))
            model.logger.scalar_summary('loss', loss_batch, e_num*user_num+i)
            loss_batch = 0.0
              
    return losses


if __name__ == '__main__':
    net = HouseCritic()
    opt = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
    criterion = nn.MSELoss()




