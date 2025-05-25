class MemoryEfficientDataset(Dataset):
    def __init__(self, tabular_data, image_data, sequence_data):
        self.tabular_keys = {item['SHA-384']: idx for idx, item in enumerate(tabular_data)}
        self.image_keys = {item['SHA-384']: idx for idx, item in enumerate(image_data)}
        self.sequence_keys = {item['SHA-384']: idx for idx, item in enumerate(sequence_data)}

        self.tabular_data = tabular_data
        self.image_data = image_data
        self.sequence_data = sequence_data

        self.common_keys = (
            set(self.tabular_keys.keys()) &
            set(self.image_keys.keys()) &
            set(self.sequence_keys.keys()))
        self.keys = list(self.common_keys)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        return {
            'tabular': self.tabular_data[self.tabular_keys[key]]['apk_features'],
            'image': self.image_data[self.image_keys[key]]['image'],
            'input_ids': self.sequence_data[self.sequence_keys[key]]['input_ids'],
            'attention_mask': self.sequence_data[self.sequence_keys[key]]['attention_mask'],
            'label': self.tabular_data[self.tabular_keys[key]]['label'],
            'SHA-384': key}