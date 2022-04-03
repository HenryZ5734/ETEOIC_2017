from data.dataset import DataSet
from torch.utils.data import DataLoader


def dataLoader(batch_size, root):
    dataset = DataSet(root=root)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=dataset.train)
    return dataloader
