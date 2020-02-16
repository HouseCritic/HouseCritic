# -*- coding: utf-8 -*-
import torch
class Env:

    PROFILE_FEATURES_DIM = 10
    SHOOPING_FEATURES_DIM = 25
    WORKSPACE_FEATURES_DIM = 8
    PROPERTY_FEATURES_DIM = 11
    VICINITY_FEATURES_DIM = 21
    REACHABILITY_FEATURES_DIM = 15

    PROFILE_EMBEDDING_DIM = 5
    SHOOPING_EMBEDDING_DIM = 10
    WORKSPACE_EMBEDDING_DIM = 5
    PROPERTY_EMBEDDING_DIM = 5
    VICINITY_EMBEDDING_DIM = 10
    REACHABILITY_EMBEDDING_DIM = 7

    FUSION_HIDDEN = 64
    FUSION_OUTPUT = 32

    META_HIDDEN_1 = 32
    META_HIDDEN_2 = 8

    GAT_HIDDEN = 64


    DROPOUT_RATE = 0.6

    


    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
