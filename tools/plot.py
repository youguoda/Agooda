# -*- coding: utf-8 -*-


from matplotlib import pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# 传递epoch，和结果路径，绘制结果图
def plot_img(epoch=58, path='./results20230213-093708.txt'):
    file = open('../' + path)
    data = file.readlines()
    train_loss = []
    train_lr = []
    dice = []
    global_correct = []
    mean_iou = []
    for i in range(1, len(data), 9):
        data_strip = data[i].strip("\n")
        train_loss.append(data_strip.split(":")[1].strip())
    train_loss = [float(n) for n in train_loss]
    # print(train_loss)

    for i in range(2, len(data), 9):
        data_strip = data[i].strip("\n")
        train_lr.append(data_strip.split(":")[1].strip())
    train_lr = [float(n) for n in train_lr]
    # print(train_lr)

    for i in range(3, len(data), 9):
        data_strip = data[i].strip("\n")
        dice.append(data_strip.split(":")[1].strip())
    dice = [float(n) for n in dice]
    # print(dice)

    for i in range(4, len(data), 9):
        data_strip = data[i].strip("\n")
        global_correct.append(data_strip.split(":")[1].strip())
    global_correct = [float(n) for n in global_correct]
    # print(global_correct)

    for i in range(7, len(data), 9):
        data_strip = data[i].strip("\n")
        mean_iou.append(data_strip.split(":")[1].strip())
    mean_iou = [float(n) for n in mean_iou]
    # print(mean_iou)

    x = np.arange(1, epoch + 1)
    plt.xticks(np.arange(0, epoch + 1, 10))  # 设置x坐标轴间距
    plt.plot(x, train_loss, color='r', label='loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title('损失变化')
    plt.legend()
    # plt.grid()
    plt.show()

    plt.xticks(np.arange(0, epoch + 1, 10))
    plt.plot(x, train_lr, color='g', label='lr')
    plt.xlabel('Epoch')
    plt.ylabel('lr')
    plt.title('学习率变化')
    plt.legend()
    # plt.grid()
    plt.show()

    plt.xticks(np.arange(0, epoch + 1, 10))
    plt.plot(x, dice, color='teal', label='dice')
    plt.xlabel('Epoch')
    plt.ylabel('dice')
    plt.title('dice分数')
    plt.legend()
    # plt.grid()
    plt.show()

    plt.xticks(np.arange(0, epoch + 1, 10))
    plt.plot(x, global_correct, color='blue', label='global_correct')
    plt.xlabel('Epoch')
    plt.ylabel('global_correct')
    plt.title('平均正确率')
    plt.legend()
    # plt.grid()
    plt.show()

    plt.xticks(np.arange(0, epoch + 1, 10))
    plt.plot(x, mean_iou, color='deeppink', label='mean_iou')
    plt.xlabel('Epoch')
    plt.ylabel('mean_iou')
    plt.title('平均Iou')
    plt.legend()
    # plt.grid()
    plt.show()


# 数据处理
def data_process(path='./results20230213-093708.txt'):
    file = open('../' + path)
    data = file.readlines()
    train_loss = []
    train_lr = []
    dice = []
    global_correct = []
    target_accuracy = []
    target_iou = []
    mean_iou = []
    data_source = []
    for i in range(1, len(data), 9):
        data_strip = data[i].strip("\n")
        train_loss.append(data_strip.split(":")[1].strip())
    train_loss = [float(n) for n in train_loss]
    data_source.append(train_loss)

    for i in range(2, len(data), 9):
        data_strip = data[i].strip("\n")
        train_lr.append(data_strip.split(":")[1].strip())
    train_lr = [float(n) for n in train_lr]
    data_source.append(train_lr)

    for i in range(3, len(data), 9):
        data_strip = data[i].strip("\n")
        dice.append(data_strip.split(":")[1].strip())
    dice = [float(n) for n in dice]
    data_source.append(dice)

    for i in range(4, len(data), 9):
        data_strip = data[i].strip("\n")
        global_correct.append(data_strip.split(":")[1].strip())
    global_correct = [float(n) for n in global_correct]
    data_source.append(global_correct)

    for i in range(5, len(data), 9):
        data_strip = data[i].strip("\n")
        if data_strip.split(":")[1].strip()[2:5] == '100':
            target_accuracy.append(data_strip.split(":")[1].strip()[11:-2])
        else:
            target_accuracy.append(data_strip.split(":")[1].strip()[10:-2])
    target_accuracy = [float(n) for n in target_accuracy]
    data_source.append(target_accuracy)

    for i in range(6, len(data), 9):
        data_strip = data[i].strip("\n")
        target_iou.append(data_strip.split(":")[1].strip()[10:-2])
    target_iou = [float(n) for n in target_iou]
    data_source.append(target_iou)

    for i in range(7, len(data), 9):
        data_strip = data[i].strip("\n")
        mean_iou.append(data_strip.split(":")[1].strip())
    mean_iou = [float(n) for n in mean_iou]
    data_source.append(mean_iou)

    return data_source


# # 根据数据进行画图
# def plot_data(data1_para, data2_para, data3_para, data4_para, data5_para, epoch=100, ylable='dice', name=' dice分数'):
#     x = np.arange(1, epoch + 1)
#     plt.xticks(np.arange(0, epoch + 1, 10))
#     plt.plot(x, data1_para, color='r', label='xx_Unet')
#     plt.plot(x, data2_para, color='b', label='Unet')
#     plt.plot(x, data3_para, color='g', label='Unet_2plus')
#     plt.plot(x, data4_para, color='orange', label='att_Unet')
#     plt.plot(x, data5_para, color='grey', label='Unet_3plus')
#     plt.xlabel('Epoch')
#     plt.ylabel(ylable)
#     plt.title(name)
#     plt.legend()
#     # plt.grid()
#     plt.show()
#
#
# # 绘制实验对比图
# def plot_compare(epoch=100, path1='results20220617-153931.txt', path2='results20220614-101316.txt',
#                  path3='results20220610-100727.txt', path4='results20220615-153710.txt', path5='results20220615-092944.txt'):
#     data1 = data_process(path1)
#     data2 = data_process(path2)
#     data3 = data_process(path3)
#     data4 = data_process(path4)
#     data5 = data_process(path5)
#     plot_data(data1[0], data2[0], data3[0], data4[0], data5[0], epoch, 'loss', '损失变化')
#     plot_data(data1[2], data2[2], data3[2], data4[2], data5[2], epoch, 'dice', 'dice分数')
#     plot_data(data1[3], data2[3], data3[3], data4[3], data5[3], epoch, 'global_correct', '准确率')
#     plot_data(data1[4], data2[4], data3[4], data4[4], data5[4], epoch, 'target_accuracy', '召回率')
#     plot_data(data1[5], data2[5], data3[5], data4[5], data5[5], epoch, 'target_iou', '目标Iou')
#     plot_data(data1[6], data2[6], data3[6], data4[6], data5[6], epoch, 'mean_iou', '平均Iou')


if __name__ == '__main__':
    # plot_compare()
    # plot_data()
    plot_img()
    print(data_process())
