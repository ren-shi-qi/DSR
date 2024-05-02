import sys
from torch.functional import norm
import torch.nn as nn
import torch
import math
from torchvision import models
import torch.nn.functional as F

class HOUR_GLASS_PROP(nn.Module):
    '''
    '''
    def __init__(self, i_cn, o_cn, _iter=1, att_cn=0, prop=False):
        super(HOUR_GLASS_PROP, self).__init__()
        self._iter = _iter
        self.prop = prop
        self.att_cn = att_cn

        dim = i_cn + att_cn
        self.conv1 = nn.Sequential(nn.Conv2d(dim, 64, 5, 1, 2),
                                  nn.LeakyReLU(0.1, inplace=True),
                                  nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 5, 1, 2),
                                  nn.LeakyReLU(0.1, inplace=True),
                                  nn.MaxPool2d(2))
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4,stride=2,padding=1,output_padding=0,bias=True),
                                  nn.LeakyReLU(0.1, inplace=True))
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(64, dim, 4,stride=2,padding=1,output_padding=0,bias=True),
                                  nn.LeakyReLU(0.1, inplace=True))
        self.reg = nn.Conv2d(dim, o_cn, 5, 1, 2)

        self._initialize_weights()

    def forward(self, fea, att=None):
        if self.att_cn > 0:
            fea = torch.cat((fea, att), 1)

        if self.prop:
            weight = torch.zeros_like(att)
            weight[att >= 0.001] = 1
            weight[att < 0.001] = 0.2
        for i in range(self._iter):
            if self.prop:
                w_fea = fea*weight
                conv1_fea = self.conv1(w_fea)
            else:
                conv1_fea = self.conv1(fea)
            conv2_fea = self.conv2(conv1_fea)
            deconv1_fea = self.deconv1(conv2_fea) + conv1_fea
            fea = self.deconv2(deconv1_fea) + fea

        self.fea = fea
        refined_smap = self.reg(fea)
        return refined_smap

    def _initialize_weights(self):
        initialize(self)

class HOUR_GLASS_PROP_DSR(nn.Module):
    '''
    '''
    def __init__(self, i_cn, o_cn, _iter=1, att_cn=0, prop=False):
        super(HOUR_GLASS_PROP_DSR, self).__init__()
        self._iter = _iter
        self.prop = prop
        self.att_cn = att_cn

        dim = i_cn + att_cn
        self.conv1 = nn.Sequential(nn.Conv2d(dim, 64, 5, 1, 2),
                                  nn.LeakyReLU(0.1, inplace=True),
                                  nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, 5, 1, 2),
                                  nn.LeakyReLU(0.1, inplace=True),
                                  nn.MaxPool2d(2))
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4,stride=2,padding=1,output_padding=0,bias=True),
                                  nn.LeakyReLU(0.1, inplace=True))
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(64, dim, 4,stride=2,padding=1,output_padding=0,bias=True),
                                  nn.LeakyReLU(0.1, inplace=True))
        self.conv3 = nn.Conv2d(257, 300, 3,stride=1,padding=1,bias=True)

        self.akm_head = AKMHead()

        # Word vectors for Standing/Sitting
        vec0 = [-0.0219, -0.0195, 0.0627, 0.0247, 0.0172, 0.2075, -0.078, -0.0536, -0.0547, 0.0106, -0.04, 0.1293, 0.0269, -0.008, -0.0241, -0.0597, -0.0309, 0.0746, -0.0811, 0.1349, -0.0266, -0.1111, -0.0959, -0.0575, 0.1274, -0.0724, -0.0228, -0.0482, -0.0135, -0.0198, -0.0755, 0.0148, 0.0159, 0.0317, 0.0184, 0.0442, 0.0462, 0.1272, 0.0784, 0.1887, 0.0577, 0.0446, -0.1689, 0.0282, -0.0575, -0.1102, 0.1221, 0.0514, -0.0413, -0.0202, -0.0338, 0.0435, -0.5966, 0.0253, 0.0549, 0.0045, 0.0184, 0.1523, 0.1471, -0.1389, -0.0133, -0.0558, 0.0761, -0.0186, -0.0942, 0.1172, 0.0526, 0.1053, 0.0127, -0.0595, 0.1186, -0.1205, -0.0018, 0.2169, 0.0288, 0.0107, 0.0474, -0.0262, 0.0619, -0.0162, 0.019, -0.0099, 0.0395, -0.164, -0.0132, -0.0979, 0.1619, -0.0267, 0.0295, -0.0202, 0.2131, 0.0488, -0.0051, -0.0741, 0.0137, 0.0847, 0.0199, 0.1065, -0.1799, -0.1458, -0.185, 0.0467, -0.0817, -0.0229, -0.1658, -0.0069, 0.162, 0.1774, -0.1561, -0.1604, -0.0405, 0.0552, 0.1224, -0.023, 0.1655, -0.1946, -0.042, -0.0023, 0.0406, -0.3122, 0.0909, -0.021, 0.0887, 0.0911, -0.0093, 0.2107, -0.0394, -0.1418, 0.0748, 0.0015, 0.0929, 0.0809, -0.0947, -0.0338, -0.0282, 0.0339, -0.075, 0.1667, -0.0468, -0.1868, 0.0734, 0.0624, -0.0091, 0.2796, 0.0873, 0.0316, 0.0296, 0.1473, 0.0603, -0.0977, 0.0821, -0.0504, -0.0165, -0.0998, -0.0258, 0.1348, -0.0835, 0.0259, -0.0814, 0.0024, -0.0275, 0.1005, -0.0103, 0.01, 0.1481, 0.0218, 0.0818, 0.0539, -0.0571, -0.0485, 0.005, 0.0915, 0.007, 0.0168, 0.1058, -0.0806, 0.2748, -0.3346, 0.0164, 0.0955, 0.1513, 0.0124, 0.102, 0.0065, -0.0989, -0.0095, -0.1508, -0.149, 0.1578, -0.0397, -0.015, 0.1837, -0.1212, 0.0003, 0.0021, 0.0878, -0.0004, 0.0338, 0.1297, 0.1542, -0.0854, 0.0207, 0.1812, -0.0451, 0.1328, -0.0238, -0.1419, -0.1375, 0.0683, -0.0316, -0.0852, 0.1101, 0.0587, 0.0628, -0.0238, 0.0284, -0.0986, -0.0124, -0.0328, -0.1432, 0.0468, -0.019, 0.0746, -0.0022, 0.0021, -0.1133, -0.0608, 0.2024, -0.1153, 0.1593, -0.0475, -0.0081, 0.3091, 0.0802, 0.1219, -0.207, 0.004, 0.064, -0.2422, 0.0302, 0.0245, 0.0371, 0.0738, -0.0133, 0.0875, 0.0248, -0.0354, 0.0311, -0.0073, 0.3672, -0.1364, -0.1001, 0.0952, -0.008, 0.0289, -0.0814, -0.1105, -0.0576, -0.1172, -0.0263, -0.0819, -0.0147, 0.0678, -0.2286, -0.153, -0.0482, 0.0458, 0.0365, -0.195, -0.2611, 0.1199, 0.0084, 0.0063, -0.083, -0.0151, 0.035, -0.0826, -0.0159, -0.1711, 0.0306, -0.073, 0.0289, 0.0415, 0.1227, 0.0681, -0.0509, -0.0407, 0.1087, -0.1913, -0.125, -0.0371, -0.0228, 0.1141, -0.0047, 0.0651, 0.0675, -0.041, 0.119, -0.0435, -0.0046]
        vec1 = [-0.0698, 0.0657, 0.0797, 0.1078, -0.0386, 0.1548, 0.0007, -0.0239, -0.0461, -0.0039, 0.0953, 0.1114, -0.0154, -0.1094, -0.1995, 0.0182, -0.1503, -0.0648, -0.0558, 0.1248, -0.0472, -0.0957, 0.0053, -0.1251, 0.0991, 0.0002, -0.1225, 0.1084, 0.0162, 0.0712, 0.0025, -0.0914, 0.0121, -0.0395, 0.0265, 0.1305, 0.173, 0.0817, 0.0372, 0.0764, -0.0387, 0.0958, -0.0379, 0.144, 0.0516, -0.2449, 0.0207, -0.0046, -0.0498, -0.0549, -0.0673, -0.2313, -0.6431, -0.1172, 0.1199, -0.199, 0.0222, 0.0506, 0.0869, -0.1606, 0.0304, -0.0513, -0.0534, -0.0952, -0.0803, 0.127, 0.1091, -0.0422, 0.0805, -0.1165, 0.0771, -0.0987, -0.0103, 0.205, 0.0661, 0.0621, 0.059, 0.0324, -0.0661, -0.1445, -0.0324, 0.1062, 0.0087, -0.2289, 0.0355, -0.0465, 0.0318, 0.0095, -0.0541, -0.0208, 0.0623, 0.1205, -0.009, 0.0298, 0.0112, 0.062, 0.0628, 0.1039, -0.1154, -0.1525, -0.1428, -0.2017, -0.0628, -0.1005, -0.2177, -0.0177, 0.2048, 0.1565, -0.0242, -0.0901, -0.1599, -0.024, 0.0788, 0.0016, 0.0297, -0.123, -0.084, 0.0288, -0.0688, -0.3015, 0.1092, -0.0595, 0.1219, 0.0323, 0.0853, 0.1507, 0.0438, -0.1347, 0.0749, 0.1002, 0.0858, -0.0088, -0.0054, 0.0011, -0.0014, 0.0746, 0.1165, 0.0544, -0.1542, -0.1177, 0.0392, 0.0367, -0.1203, 0.1564, 0.0868, 0.1091, 0.0362, 0.0518, -0.0054, -0.0624, 0.0808, -0.1181, -0.0004, -0.0861, -0.0777, 0.1585, -0.0159, 0.0674, -0.1354, 0.0362, -0.0098, 0.0885, -0.0321, -0.1037, 0.1267, 0.0354, -0.0073, 0.0818, 0.0594, -0.0515, 0.0308, 0.0015, 0.103, 0.0517, 0.2073, -0.1619, 0.2866, -0.2622, 0.0297, 0.0044, 0.0499, 0.0635, 0.0, 0.1202, 0.0527, -0.0855, -0.1791, -0.1535, 0.2964, -0.0829, -0.0906, 0.2045, 0.0085, 0.1541, 0.0675, 0.146, 0.0758, 0.0432, 0.0893, -0.017, -0.1112, -0.0256, 0.1067, -0.0464, 0.0187, 0.0044, -0.0535, -0.1669, -0.0309, 0.1324, -0.1188, -0.0302, 0.1028, -0.0014, 0.0416, -0.1368, -0.2871, 0.0393, 0.1242, 0.0377, 0.1612, -0.1231, 0.1003, 0.0389, 0.0134, -0.0593, -0.1021, 0.0754, 0.0789, 0.2652, -0.0097, -0.0402, 0.2298, -0.0444, 0.0016, -0.0894, -0.0632, 0.0352, -0.2305, 0.0435, -0.037, -0.0171, 0.0918, -0.1384, 0.221, -0.0123, -0.0203, -0.0611, -0.0534, 0.3978, -0.0864, -0.1886, 0.2037, 0.0926, 0.0677, -0.1161, -0.1094, -0.08, 0.0056, -0.0981, -0.2157, 0.0674, 0.1146, -0.1391, -0.1144, -0.0324, 0.0842, 0.0257, -0.2472, -0.1406, 0.0566, 0.0859, 0.0168, -0.0664, 0.0222, 0.001, 0.1161, 0.015, -0.0479, 0.1375, -0.1128, 0.0819, 0.1934, 0.0781, 0.1462, 0.0444, -0.1604, 0.0038, -0.0141, -0.0838, -0.0411, -0.0067, 0.0663, -0.0621, 0.0729, 0.0739, 0.005, 0.0971, -0.0545, 0.0197]
        
        self.vec0 = torch.tensor(vec0).cuda()
        self.vec1 = torch.tensor(vec1).cuda()

        # if set as trainable
        # self.vec0 = nn.parameter(torch.tensor(vec0).cuda())
        # self.vec1 = nn.parameter(torch.tensor(vec1).cuda())
        
        self.o_cn = o_cn
        self._initialize_weights()

    def forward(self, fea, att=None):
        if self.att_cn > 0:
            fea = torch.cat((fea, att), 1)

        if self.prop:
            weight = torch.zeros_like(att)
            weight[att >= 0.001] = 1
            weight[att < 0.001] = 0.2
        for i in range(self._iter):
            if self.prop:
                w_fea = fea*weight
                conv1_fea = self.conv1(w_fea)
            else:
                conv1_fea = self.conv1(fea)
            conv2_fea = self.conv2(conv1_fea)
            deconv1_fea = self.deconv1(conv2_fea) + conv1_fea
            fea = self.deconv2(deconv1_fea) + fea

        #stacked feature
        stacked_features = []
        stacked_features.append(conv2_fea)
        stacked_features.append(deconv1_fea)
        stacked_features.append(fea)
        stacked_features = self.split_feats(stacked_features)
        wv_fea = self.conv3(stacked_features)
        H = stacked_features.shape[-2]
        W = stacked_features.shape[-1]

        #AKM
        self.kernel_pred = self.akm_head(stacked_features)

        #WVM
        self.class0_weight = self.vec0.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        self.class1_weight = self.vec1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        bg = torch.zeros([1,1,H,W]).cuda()
        persons = torch.zeros([1,1,H,W]).cuda()
        class0 = torch.zeros([1,1,H,W]).cuda()
        class1 = torch.zeros([1,1,H,W]).cuda()
        
        # foreground and background
        for i in range(fea.shape[1]): 
            bg += F.conv2d(fea[:,i,:,:].unsqueeze(0), self.kernel_pred[:,int(i/8),:,:].unsqueeze(0), stride=1, padding=2)
            persons += F.conv2d(fea[:,i,:,:].unsqueeze(0), self.kernel_pred[:,int(i/8) + 9,:,:].unsqueeze(0), stride=1, padding=2)

        # word vector convolution  
        class0_ = F.conv2d(wv_fea, self.class0_weight, stride=1, padding=0)
        class1_ = F.conv2d(wv_fea, self.class1_weight, stride=1, padding=0)

        # class0 and class1 
        class0 = F.conv2d(persons, self.kernel_pred[:,18,:,:].unsqueeze(0), stride=1, padding=2)
        class1 = F.conv2d(persons, self.kernel_pred[:,19,:,:].unsqueeze(0), stride=1, padding=2)

        # fusion ration lambda
        frat = 0.4
        persons0 = frat * class0_ + (1-frat) * class0
        persons1 = frat * class1_ + (1-frat) * class1

        refined_smap = torch.cat((persons0,persons1,bg),1)

        return refined_smap


    def _initialize_weights(self):
        initialize(self)
    
    @staticmethod
    def split_feats(feats):
        temp = torch.cat([F.interpolate(feats[0], scale_factor=4, mode='bilinear'),F.interpolate(feats[1], scale_factor=2, mode='bilinear')],dim=1)
        temp = torch.cat([temp,feats[2]],dim=1)
        return (temp)

class AKMHead(nn.Module):
    def __init__(self):

        super().__init__()

        # num_kernels equals to 2*k+2
        self.num_kernels = 20
        self.instance_in_channels = 257
        self.instance_channels = 257

        norm = ""
        tower = []
        num_convs, use_deformable, use_coord = 4, False, False
        for i in range(num_convs):
            conv_func = nn.Conv2d
            if i == 0:
                if use_coord:
                    chn = self.instance_in_channels + 2
                else:
                    chn = self.instance_in_channels
            else:
                chn = self.instance_channels

            tower.append(conv_func(
                    chn, self.instance_channels,
                    kernel_size=3, stride=1,
                    padding=1, bias=norm is None
            ))
            if norm == "GN":
                tower.append(nn.GroupNorm(32, self.instance_channels))
            tower.append(nn.ReLU(inplace=True))
        self.add_module('kernel_tower',
                        nn.Sequential(*tower))

        self.kernel_pred = nn.Conv2d(
            self.instance_channels, self.num_kernels,
            kernel_size=3, stride=1, padding=1
        )

        for modules in [
            self.kernel_tower,
            self.kernel_pred,
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if l.bias is not None:
                        nn.init.constant_(l.bias, 0)


    def forward(self, features):
        """
        Returns:
            pass
        """
        kernel_pred = []
        
        # kernel feature.
        kernel_feat = features

        # kernel generation
        kernel_feat = self.kernel_tower(kernel_feat)
        kernel_feat = F.interpolate(kernel_feat, size = [5,5])
        kernel_pred = self.kernel_pred(kernel_feat)

        return kernel_pred

class CSRNet_TWO(nn.Module):
    def __init__(self, i_cn, o_cn):
        super(CSRNet_TWO, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512,256,128,64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat,in_channels = 512,dilation = True)
        self.backend_seg = make_layers(self.backend_feat,in_channels = 512,dilation = True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.output_layer_seg = nn.Conv2d(64, o_cn, kernel_size=1)
        self._initialize_weights()
        if True:
            mod = models.vgg16(pretrained = True)
            fs = self.frontend.state_dict()
            ms = mod.state_dict()
            for key in fs:
                fs[key] = ms['features.'+key]
            self.frontend.load_state_dict(fs)
        else:
            print("Don't pre-train on ImageNet")

    def forward(self,x):
        x = self.frontend(x)
        self.smap_fea = self.backend_seg(x)
        self.dmap_fea = self.backend(x)
        x = self.output_layer(self.dmap_fea)
        smap = self.output_layer_seg(self.smap_fea)
        #x = F.interpolate(x, scale_factor=8)
        return x, smap

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels = 3,norm=False,dilation = False, dropout=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def initialize(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
