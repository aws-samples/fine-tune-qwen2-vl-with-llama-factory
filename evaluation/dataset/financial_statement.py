from datasets import load_dataset
from torch.utils.data import Dataset

from .utils import pil_to_base64


class FinancialStatementDataset(Dataset):

    def __init__(self,
                 data_path='apoidea/financial-statement-table-html',
                 split='test'):
        super().__init__()
        self.data = load_dataset(data_path, split=split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return int(idx), pil_to_base64(sample['image']), sample['html_table']


if __name__ == '__main__':
    dataset = FinancialStatementDataset(
        data_path='apoidea/financial-statement-table-html')
    print(len(dataset))
    image_id, image, html = dataset[0]
    breakpoint()
