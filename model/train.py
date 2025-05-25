def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad()

    for i, batch in enumerate(tqdm(dataloader, desc="Training")):
        tabular = batch['tabular'].float().to(DEVICE, non_blocking=True)
        image = batch['image'].float().to(DEVICE, non_blocking=True)
        input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
        attention_mask = batch['attention_mask'].to(DEVICE, non_blocking=True)
        labels = batch['label'].to(DEVICE, non_blocking=True)

        with torch.amp.autocast(device_type='cuda'):
            outputs = model(tabular, image, input_ids, attention_mask)
            loss = criterion(outputs, labels) / GRADIENT_ACCUMULATION_STEPS

        loss.backward()

        if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (i + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            gc.collect()

        running_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / len(dataloader), correct / total

def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            tabular = batch['tabular'].float().to(DEVICE)
            image = batch['image'].float().to(DEVICE)
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            outputs = model(tabular, image, input_ids, attention_mask)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()

    return (running_loss / len(dataloader), correct / total, all_preds, all_labels)