def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            tabular = batch['tabular'].float().to(DEVICE)
            image = batch['image'].float().to(DEVICE)
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            outputs = model(tabular, image, input_ids, attention_mask)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['benign', 'adware', 'banking', 'riskware', 'sms']))

    return {
        'loss': running_loss / len(dataloader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'preds': all_preds,
        'labels': all_labels
    }

def evaluate_checkpoint(checkpoint_path, test_loader):
    model = OptimizedMultimodalModel(
        tabular_input_dim=len(tabular_dataset_train[0]['apk_features']),
        num_classes=NUM_CLASSES).to(DEVICE)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    criterion = nn.CrossEntropyLoss()
    results = evaluate(model, test_loader, criterion)

    print("\nFinal Test Metrics:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-Score: {results['f1']:.4f}")

    return results

evaluate_checkpoint("checkpoint_epoch_15.pth", test_loader)