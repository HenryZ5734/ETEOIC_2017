import os
from torchvision import transforms


def tensor2image(X_hat, epoch, path):

    # 图像转换
    toPIL = transforms.ToPILImage()
    for i in range(X_hat.shape[0]):
        image = toPIL(X_hat[i].clamp(0., 1.))       # 将X_hat张量的值规约到0，1之间
        image.save(os.path.join(path, str(epoch) + '-' + str(i) + '.bmp'))

