"""
数据集
"""
import torch.utils.data as tdata
import cv2


class ContainData(tdata.Dataset):
    """
    数据集
    """
    def __init__(self, data_file, transform):
        super(ContainData, self).__init__()
        self.transform = transform
        with open(data_file, 'r') as f:
            lines = f.readlines()
            datas = []
            for line in lines:
                arr = line.strip().split(',')
                datas.append({
                    'target': arr[0],
                    'origin': arr[1],
                    'point': [float(item) for item in arr[2:]]
                })
            self.datas = datas

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        target = cv2.imread(self.datas[idx]['target'])
        origin = cv2.imread(self.datas[idx]['origin'])
        point = self.datas[idx]['point']
        # 变换
        target, origin = self.transform(target, origin)
        # 转为 rgb格式
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        origin = cv2.cvtColor(origin, cv2.COLOR_BGR2RGB)
        return target, origin, point
