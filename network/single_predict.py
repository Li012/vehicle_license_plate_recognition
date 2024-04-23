'''
    功能：按着路径，导入单张图片做预测
    作者： Leo在这

'''
from torchvision.models import resnet18
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable


class detect_model(object):
    def __init__(self):
        # 如果显卡可用，则用显卡进行训练
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")

        '''
            加载模型与参数
        '''

        # 加载模型
        self.model = resnet18(pretrained=False, num_classes=65).to(self.device)  # 43.6%

        # 加载模型参数
        if self.device == "cpu":
            # 加载模型参数
            self.model.load_state_dict(torch.load("./network/model_resnet18.pth", map_location=torch.device('cpu')))
        else:
            self.model.load_state_dict(torch.load("./network/model_resnet18.pth"))
        # if self.device == "cpu":
        #     # 加载模型参数
        #     self.model.load_state_dict(torch.load("./model_resnet18.pth", map_location=torch.device('cpu')))
        # else:
        #     self.model.load_state_dict(torch.load("./model_resnet18.pth"))

    def detect(self, img):
        # img = Image.open(img_path)  # 打开图片
        img = img.convert('RGB')  # 转换为RGB 格式
        img = padding_black(img)
        # print(type(img))
        img_tensor = val_tf(img)
        # print(type(img_tensor))
        # 增加batch_size维度
        img_tensor = Variable(torch.unsqueeze(img_tensor, dim=0).float(), requires_grad=False).to(self.device)

        '''
            数据输入与模型输出转换
        '''
        self.model.eval()
        with torch.no_grad():
            output_tensor = self.model(img_tensor)
            # print(output_tensor)

            # 将输出通过softmax变为概率值
            output = torch.softmax(output_tensor, dim=1)
            # print(output)
            #
            # 输出可能性最大的那位
            pred_value, pred_index = torch.max(output, 1)
            # print(pred_value)
            # print(pred_index)
            #
            # 将数据从cuda转回cpu
            if torch.cuda.is_available() == False:
                pred_value = pred_value.detach().cpu().numpy()
                pred_index = pred_index.detach().cpu().numpy()
            #
            # print(pred_value)
            # print(pred_index)
            #
            # 增加类别标签
            classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H", "J",
                       "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "川", "鄂", "赣", "甘",
                       "贵", ",桂", "黑", "沪", "冀", "津", "京", "吉", "辽", "鲁", "蒙", "闽", "宁", "青", "琼", "陕", "苏", "晋", "皖",
                       "湘", "新", "豫", "渝", "粤", "云", "藏", "浙"]
            #
            # # result = "预测类别为： " + str(classes[pred_index[0]]) + " 可能性为: " + str(pred_value[0] * 100) + "%"
            #
            print("预测类别为： ", classes[pred_index[0]], " 可能性为: ", pred_value[0] * 100, "%")



'''
    加载图片与格式转化
'''
# img_path = './ann/9/64-1.jpg'

# 图片标准化
transform_BZ = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],  # 取决于数据集
    std=[0.5, 0.5, 0.5]
)

val_tf = transforms.Compose([
    # transforms.Resize(224),
    transforms.Resize(20),
    transforms.ToTensor(),
    transform_BZ  # 标准化操作
])


def padding_black(img):  # 如果尺寸太小可以扩充
    w, h = img.size
    # scale = 224. / max(w, h)
    scale = 20. / max(w, h)
    img_fg = img.resize([int(x) for x in [w * scale, h * scale]])
    size_fg = img_fg.size
    # size_bg = 224
    size_bg = 20
    img_bg = Image.new("RGB", (size_bg, size_bg))
    img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                          (size_bg - size_fg[1]) // 2))
    img = img_bg
    return img


if __name__ == "__main__":
    path = './pic/pic0.jpg'
    model = detect_model()
    model.detect(path)
