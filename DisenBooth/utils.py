import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from accelerate.tracking import GeneralTracker, on_main_process
import os
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def make_grid(pil_images, nrow,padding=2):
    """
    将多个PIL图像合并为一个大的图像网格。

    :param pil_images: PIL Image对象的列表。
    :param nrow: 每行显示图像的数量。
    :param padding: 图像之间的间距。
    :return: 一个PIL Image对象，表示图像网格。
    """
    # 确定网格的尺寸
    single_width, single_height = pil_images[0].size
    ncol = int((len(pil_images) + nrow - 1) / nrow)  # 计算需要的列数
    total_width = nrow * single_width + (nrow - 1) * padding
    total_height = ncol * single_height + (ncol - 1) * padding

    # 创建新的PIL图像作为网格的底图
    grid_image = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))

    # 将单个图像拼接到网格的适当位置
    for i, image in enumerate(pil_images):
        row = int(i / nrow)
        col = i % nrow
        xoffset = col * (single_width + padding)
        yoffset = row * (single_height + padding)
        grid_image.paste(image, (xoffset, yoffset))

    return grid_image


# 0. 自定义追踪器
class MyCustomTracker(GeneralTracker):
    """
    my custom `Tracker` class that supports `tensorboard`. Should be initialized at the start of your script.

    Args:
        run_name (`str`):
            The name of the experiment run
        logging_dir (`str`, `os.PathLike`):
            Location for TensorBoard logs to be stored.
        kwargs:
            Additional key word arguments passed along to the `tensorboard.SummaryWriter.__init__` method.
    """

    name = "tensorboard"
    requires_logging_directory = True

    @on_main_process
    def __init__(self, run_name: str, logging_dir: Union[str, os.PathLike],
                 **kwargs):
        super().__init__()
        self.run_name = run_name
        self.logging_dir = os.path.join(logging_dir, run_name)
        self.writer = SummaryWriter(self.logging_dir, **kwargs)

    @property
    def tracker(self):
        return self.writer

    @on_main_process
    def add_scalar(self, tag, scalar_value, **kwargs):
        self.writer.add_scalar(tag=tag, scalar_value=scalar_value, **kwargs)

    @on_main_process
    def add_text(self, tag, text_string, **kwargs):
        self.writer.add_text(tag=tag, text_string=text_string, **kwargs)

    @on_main_process
    def add_figure(self, tag, figure, **kwargs):
        self.writer.add_figure(tag=tag, figure=figure, **kwargs)
    @on_main_process
    def add_image(self, tag, img_tensor,dataformats, **kwargs):
        self.writer.add_image(tag=tag, img_tensor=img_tensor,dataformats=dataformats, **kwargs)





if __name__ == "__main__":
    ## 自定义跟踪器测试
    # # 1. 初始化 Accelerator，并使用自定义的tensorbaord追踪器
    # mycustomtracker = MyCustomTracker(run_name='test', logging_dir='testfolder')
    # accelerator = Accelerator(log_with=mycustomtracker)


    # # 2. 定义一个简单的模型和优化器
    # class SimpleModel(nn.Module):

    #     def __init__(self):
    #         super(SimpleModel, self).__init__()
    #         self.fc = nn.Linear(10, 2)

    #     def forward(self, x):
    #         return self.fc(x)


    # model = SimpleModel()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)

    # # 3. 定义虚拟数据
    # x = torch.randn(100000, 10)
    # y = (torch.randn(100000) > 0.5).long()
    # dataset = TensorDataset(x, y)
    # loader = DataLoader(dataset, batch_size=2048, shuffle=True)

    # # 4.画图demo
    # fig1, ax1 = plt.subplots()
    # xplot1 = np.array([1, 2, 3, 4, 5])
    # yplot1 = np.array([1, 2, 3, 4, 5])
    # ax1.plot(xplot1, yplot1)

    # fig2, ax2 = plt.subplots()
    # xplot2 = np.array([1, 2, 3, 4, 5])
    # yplot2 = np.array([5, 4, 3, 2, 1])
    # ax2.plot(xplot2, yplot2)
    # ref_im = Image.open("/mnt/workspace/workgroup_share/lhn/diffusion_basics/DisenBooth/dog7/00.jpg")
    # image_np = np.array(ref_im)

    # mycustomtracker.add_image(tag="validation images", img_tensor = image_np,dataformats="HWC",global_step=1)

    # mycustomtracker.add_figure(tag="test_plot", figure=fig1, global_step=1, close=True, walltime=None)
    # mycustomtracker.add_figure(tag="test_plot", figure=fig2, global_step=2, close=True, walltime=None)

    # # 5.添加textdemo
    # mycustomtracker.add_text(tag="test_text", text_string='This is a test string')

    # # 5. 使用 accelerator.prepare 函数准备模型和优化器
    # model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    # # 6. 训练循环
    # iter = 1
    # for epoch in range(10):
    #     for batch in loader:
    #         inputs, targets = batch

    #         outputs = model(inputs)
    #         loss = nn.CrossEntropyLoss()(outputs, targets)

    #         optimizer.zero_grad()

    #         accelerator.backward(loss)

    #         optimizer.step()
    #         print(f"Loss: {loss.item()}")

    #         mycustomtracker.add_scalar(tag='training loss',
    #                                 scalar_value=loss.item(),
    #                                 global_step=iter)
    #         iter += 1

    # print("Training completed!")
    ## make_grid测试
    root_dir = "/mnt/workspace/workgroup_share/lhn/diffusion_basics/DisenBooth/dog7"
    image_files = os.listdir(root_dir)
    # 示例使用
    images = [Image.open(f"{root_dir}/{file}") for file in image_files]  # 'image_files' 应为图像文件的路径列表
    grid = make_grid(images, nrow=3)
    grid.save("train.png")
