import torch
import torch.nn as nn
from common.models import ClipVisionExtractor, Extractor, BertExtractor, ResNetExtractor, EfficientNetExtractor

__all__ = [ 'BaseRingExtractor',
            'Base3DObjectRingsExtractor']

class TransEncoderBlock(nn.Module):
    def __init__(self, feature_dim, num_heads, dropout, hidden_dim=None, batch_first=False):
        super().__init__()
        self.mha = nn.MultiheadAttention(feature_dim, num_heads, dropout, batch_first=batch_first)
        self.layernorm1 = nn.LayerNorm(feature_dim)
        self.layernorm2 = nn.LayerNorm(feature_dim)
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
    
    def forward(self, x):
        '''
            (B, V, D) --> (B, V, D)
        '''
        q = x
        mha_out, _ = self.mha(q, q, q)
        x = self.layernorm1(x + mha_out)
        fc_out = self.fc(x)
        out = self.layernorm2(x + fc_out)
        return out

class BaseRingExtractor(nn.Module):
    def __init__(self, view_cnn_backbone='resnet50', view_seq_embedder='bilstm', hidden_dim=512):
        super().__init__()

        if view_cnn_backbone.startswith('resnet'):
            self.cnn = ResNetExtractor(view_cnn_backbone)
        elif view_cnn_backbone.startswith('efficientnet'):
            self.cnn = EfficientNetExtractor(view_cnn_backbone)
        elif view_cnn_backbone.startswith('openai'):
            self.cnn = ClipVisionExtractor(view_cnn_backbone)
        else:
            raise NotImplementedError
        
        self.cnn_feature_dim = self.cnn.feature_dim  # D

        if view_seq_embedder == 'bilstm':
            # original baseline
            self.feature_dim = 2 * hidden_dim  # D'
            self.embedder = nn.LSTM(self.cnn_feature_dim,
                                hidden_dim,
                                batch_first=True,
                                bidirectional=True)
        elif view_seq_embedder == 'mha':
            self.feature_dim = self.cnn_feature_dim     # D'
            self.embedder = TransEncoderBlock(self.cnn_feature_dim, num_heads=4, dropout=0.0, hidden_dim=2*self.feature_dim, batch_first=True)

    def get_embedding(self, x):
        # x: [B, V, C, H, W]
        B, V, C, H, W = x.size()
        x = x.reshape(B*V, C, H, W)  # B*V, C, H, W
        x = self.cnn.get_embedding(x)  # B*V, D
        x = x.reshape(B, V, self.cnn_feature_dim)  # B, V, D
        if isinstance(self.embedder, nn.LSTM):
            x, _ = self.embedder(x)  # B, V, D'
        else:
            x = self.embedder(x) # B, V, D'
        x = x.mean(1)  # B, D'
        return x
    

class Base3DObjectRingsExtractor(nn.Module):
    def __init__(self, num_rings, view_cnn_backbone='resnet50', view_seq_embedder='bilstm', num_mhas=1, num_heads=4, dropout=0.0, reverse=False):
        super().__init__()
        self.kwargs = {'num_rings': num_rings, 'view_cnn_backbone': view_cnn_backbone, 'view_seq_embedder': view_seq_embedder,'num_mhas':num_mhas,'num_heads':num_heads,'dropout':dropout}
        self.reverse = reverse
        if reverse:
            num_rings = 12
        self.ring_exts = nn.ModuleList([
            BaseRingExtractor(view_cnn_backbone, view_seq_embedder)
            for _ in range(num_rings)
        ])
        self.view_feature_dim = self.ring_exts[0].feature_dim  # D
        self.feature_dim = self.view_feature_dim  # D'

        if num_mhas == 1:
            # original baseline
            self.attn = nn.MultiheadAttention(self.feature_dim, num_heads, dropout)
        else:
            encoders = []
            for _ in range(num_mhas):
                encoders.append(TransEncoderBlock(self.feature_dim, hidden_dim=1024, num_heads=num_heads, dropout=dropout))
            self.attn = nn.Sequential(*encoders)

    def forward(self, x):
        # x: B, R, V, C, H, W
        if self.reverse:
            x = x.transpose(1, 2)
        x = torch.cat([
            ring_ext.get_embedding(x[:, i]).unsqueeze(1)
            for i, ring_ext in enumerate(self.ring_exts)
        ], dim=1)  # B, R, D
        x = x.transpose(0, 1)  # R, B, D
        if isinstance(self.attn, nn.MultiheadAttention):
            x, _ = self.attn(x, x, x)  # R, B, D
        else:
            x = self.attn(x)  # R, B, D
        x = x.mean(0)  # B, D
        return x

    def get_embedding(self, x):
        # x: B, R, V, C, H, W
        return self.forward(x)

    
if __name__ == "__main__":
    from dataset import *
    ds = SHREC23_Rings_RenderOnly_ImageQuery(
        'data/train.csv', 'data/ANIMAR_Preliminary_Data/generated_models', 'data/ANIMAR_Preliminary_Data/Sketch_Query', [0, 1, 2])
    ds2 = SHREC23_Rings_RenderOnly_TextQuery(
        'data/train.csv', 'data/ANIMAR_Preliminary_Data/generated_models', 'data/ANIMAR_Preliminary_Data/Sketch_Query', [0, 1, 2])
    dl = data.DataLoader(ds, batch_size=2, collate_fn=ds.collate_fn)
    dl2 = data.DataLoader(ds2, batch_size=2, collate_fn=ds2.collate_fn)


    obj_extractor = Base3DObjectRingsExtractor(
        num_heads=4,
        dropout=0.1,
        num_rings=3,
    )
    obj_embedder = MLP(obj_extractor)
    img_extractor = ResNetExtractor()
    img_embedder = MLP(img_extractor)
    txt_extractor = BertExtractor()
    txt_embedder = MLP(txt_extractor)

    batch = next(iter(dl))
    ring_inputs = batch['object_ims']
    img_query = batch['query_ims']

    ring_outputs = obj_embedder(ring_inputs)
    print(ring_outputs.shape)

    img_query_outputs = img_embedder(img_query)
    print(img_query_outputs.shape)

    batch = next(iter(dl2))
    txt_query = batch['tokens']
    txt_query_outputs = txt_embedder(txt_query)
    print(txt_query_outputs.shape)