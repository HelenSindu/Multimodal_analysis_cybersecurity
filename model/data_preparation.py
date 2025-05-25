def prepare_tabular_data(tabular_data_train, tabular_data_test):
    mapping_for_labels = {'adware': 1, 'banking': 2, 'riskware': 3, 'sms': 4, 'benign': 0}

    tabular_data_train['Class'] = tabular_data_train['Class'].map(mapping_for_labels).astype(int)
    tabular_data_test['Class'] = tabular_data_test['Class'].map(mapping_for_labels).astype(int)

    for column in tabular_data_train.columns:
        if column not in ['apk_name', 'Class']:
            tabular_data_train[column] = pd.to_numeric(tabular_data_train[column], errors='coerce')
            tabular_data_test[column] = pd.to_numeric(tabular_data_test[column], errors='coerce')

    tabular_data_train = tabular_data_train.fillna(0)
    tabular_data_test = tabular_data_test.fillna(0)

    tabular_dataset_train = []
    for _, row in tabular_data_train.iterrows():
        features = row.drop(['apk_name', 'Class']).values.astype(np.float32)
        tabular_dataset_train.append({
            'SHA-384': row['apk_name'],
            'label': torch.tensor(row['Class'], dtype=torch.long),
            'apk_features': torch.tensor(features, dtype=torch.float32)
        })

    tabular_dataset_test = []
    for _, row in tabular_data_test.iterrows():
        features = row.drop(['apk_name', 'Class']).values.astype(np.float32)
        tabular_dataset_test.append({
            'SHA-384': row['apk_name'],
            'label': torch.tensor(row['Class'], dtype=torch.long),
            'apk_features': torch.tensor(features, dtype=torch.float32)
        })

    return tabular_dataset_train, tabular_dataset_test

def prepare_image_data(image_path_train, image_path_test):
    mapping_for_labels = {'adware': 1, 'banking': 2, 'riskware': 3, 'sms': 4, 'benign': 0}
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def load_images(image_path):
        images = []
        labels = []
        sha384_list = []
        for folder_name, label in mapping_for_labels.items():
            folder = os.path.join(image_path, folder_name)
            if os.path.exists(folder):
                for img_name in os.listdir(folder):
                    img_path = os.path.join(folder, img_name)
                    try:
                        image = Image.open(img_path).convert('RGB')
                        images.append(transform(image))
                        labels.append(label)
                        sha384_list.append(os.path.splitext(img_name)[0])
                    except:
                        continue
        return images, labels, sha384_list

    images_train, labels_train, sha384_train = load_images(image_path_train)
    images_test, labels_test, sha384_test = load_images(image_path_test)

    image_dataset_train = [{
        'SHA-384': sha384,
        'label': torch.tensor(label, dtype=torch.long),
        'image': image
    } for sha384, label, image in zip(sha384_train, labels_train, images_train)]

    image_dataset_test = [{
        'SHA-384': sha384,
        'label': torch.tensor(label, dtype=torch.long),
        'image': image
    } for sha384, label, image in zip(sha384_test, labels_test, images_test)]

    return image_dataset_train, image_dataset_test

def prepare_sequence_data(sequence_data_train, sequence_data_test):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def tokenize_data(data, tokenizer):
        encodings = tokenizer(
            data['Label'].tolist(),
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        return encodings

    sequence_encodings_train = tokenize_data(sequence_data_train, tokenizer)
    sequence_encodings_test = tokenize_data(sequence_data_test, tokenizer)

    mapping_for_labels = {'adware': 1, 'banking': 2, 'riskware': 3, 'sms': 4, 'benign': 0}
    sequence_data_train['Class'] = sequence_data_train['Class'].map(mapping_for_labels)
    sequence_data_test['Class'] = sequence_data_test['Class'].map(mapping_for_labels)

    def create_sequence_dataset(data, encodings):
        dataset = []
        for i in range(len(data)):
            dataset.append({
                'SHA-384': data.iloc[i]['GMLnames'],
                'input_ids': encodings['input_ids'][i],
                'attention_mask': encodings['attention_mask'][i],
                'label': torch.tensor(data.iloc[i]['Class'], dtype=torch.long)
            })
        return dataset

    sequence_dataset_train = create_sequence_dataset(sequence_data_train, sequence_encodings_train)
    sequence_dataset_test = create_sequence_dataset(sequence_data_test, sequence_encodings_test)

    return sequence_dataset_train, sequence_dataset_test