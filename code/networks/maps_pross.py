# 利用tensorboardX可视化特征图
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

writer = SummaryWriter(log_dir='logs', comment=x)
# for i, data in enumerate(trainloader, 0):
#     # 获取训练数据
#     inputs, labels = data
#     inputs, labels = inputs.to(device), labels.to(device)
#     x = inputs[0].unsqueeze(0)  # x 在这里呀
#     break

# img_grid = vutils.make_grid(x, normalize=True, scale_each=True, nrow=2)

# net.eval()

# for name, layer in net._modules.items():
#     # 为fc层预处理x
#     x = x.view(x.size(0), -1) if 'fc' in name else x
#     print(x.size())
#
#     x = layer(x)
#     print(f'{name}')

    # 查看卷积层的特征图
# if 'layer' in name or 'conv' in name:
#         x1 = x.transpose(0, 1)  # C，B, H, W ---> B，C, H, W
# img_grid = vutils.make_grid(x, normalize=True, scale_each=True, nrow=4)
# writer.add_image(img_grid, global_step=0)