class LiteDNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 128], dropout=0.3):
        super(LiteDNN, self).__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)])
            prev_dim = hidden_dim

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class LiteCNN(nn.Module):
    def __init__(self, in_channels=3, base_channels=8):
        super(LiteCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)))

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)

class LiteBERT(nn.Module):
    def __init__(self, hidden_size=64):
        super(LiteBERT, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')

        for param in self.distilbert.parameters():
            param.requires_grad = False

        for layer in self.distilbert.transformer.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Linear(self.distilbert.config.dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2))

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True)
        return self.classifier(outputs.last_hidden_state[:, 0, :])

class OptimizedMultimodalModel(nn.Module):
    def __init__(self, tabular_input_dim, num_classes=5):
        super(OptimizedMultimodalModel, self).__init__()
        self.dnn = LiteDNN(tabular_input_dim)
        self.cnn = LiteCNN()
        self.bert = LiteBERT()

        self.fusion = nn.Sequential(
            nn.Linear(128 + 16 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes))

    def forward(self, tabular, image, input_ids, attention_mask):
        tabular_out = self.dnn(tabular)
        image_out = self.cnn(image)
        bert_out = self.bert(input_ids, attention_mask)
        combined = torch.cat([tabular_out, image_out, bert_out], dim=1)
        return self.fusion(combined)