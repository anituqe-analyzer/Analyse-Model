import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model import AuctionAuthenticityModel
from dataset_loader import AuctionDatasetFromJSON, get_transforms
import json

def train_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    progress_bar = tqdm(loader, desc=f"Epoch {epoch} [TRAIN]")
    
    for batch in progress_bar:
        images = batch['image'].to(device)
        texts = batch['text']
        labels = batch['label'].to(device)
        cat_labels = batch.get('category', None)
        if cat_labels is not None:
            cat_labels = cat_labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images, texts)
        loss, losses = model.compute_loss(outputs, auth_labels=labels, cat_labels=cat_labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=f'{loss.item():.4f}')
    
    return total_loss / len(loader)

def validate(model, loader, device, epoch):
    model.eval()
    all_preds = []
    all_labels = []
    all_cat_preds = []
    all_cat_labels = []
    total_loss = 0
    
    with torch.no_grad():
        progress_bar = tqdm(loader, desc=f"Epoch {epoch} [VAL]")
        for batch in progress_bar:
            images = batch['image'].to(device)
            texts = batch['text']
            labels = batch['label'].to(device)
            cat_labels = batch.get('category', None)
            if cat_labels is not None:
                cat_labels = cat_labels.to(device)
            
            outputs = model(images, texts)
            loss, losses = model.compute_loss(outputs, auth_labels=labels, cat_labels=cat_labels)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs['auth_probs'], dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            # Category metrics (only for valid labels >=0)
            if cat_labels is not None:
                mask = (cat_labels >= 0)
                if mask.sum().item() > 0:
                    cat_preds = torch.argmax(outputs['cat_probs'][mask], dim=1).cpu().numpy()
                    cat_true = cat_labels[mask].cpu().numpy()
                    all_cat_preds.extend(cat_preds)
                    all_cat_labels.extend(cat_true)
    
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cat_acc = None
    if len(all_cat_labels) > 0:
        cat_acc = accuracy_score(all_cat_labels, all_cat_preds)
    
    return {
        'loss': total_loss / len(loader),
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'cat_accuracy': cat_acc
    }

def main():
    # Konfiguracja
    BATCH_SIZE = 4
    EPOCHS = 10
    LEARNING_RATE = 2e-5
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🖥️  Device: {DEVICE}")
    print(f"📦 Batch size: {BATCH_SIZE}")
    print(f"📚 Epochs: {EPOCHS}")
    
    # Załaduj dataset
    print("\n📥 Ładowanie datasetu...")
    dataset = AuctionDatasetFromJSON(
        json_path='../dataset/dataset.json',
        root_dir='../dataset/raw_data',
        transform=get_transforms()
    )
    
    print(f"✓ {len(dataset)} aukcji załadowanych")
    
    # Split: 80% train, 20% val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"  - Train: {len(train_dataset)}")
    print(f"  - Val: {len(val_dataset)}")
    
    # Model
    print("\n🧠 Inicjalizacja modelu...")
    model = AuctionAuthenticityModel(device=DEVICE).to(DEVICE)
    print(f"✓ Model gotowy ({model.count_parameters():,} parametrów)")
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("\n🚀 Rozpoczynam trening...\n")
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': [],
        'val_cat_accuracy': []
    }
    
    for epoch in range(EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE, epoch+1)
        
        # Validate
        val_metrics = validate(model, val_loader, DEVICE, epoch+1)
        
        # Log
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_cat_accuracy'].append(val_metrics.get('cat_accuracy'))
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_metrics['loss']:.4f}")
        print(f"  Val Acc:    {val_metrics['accuracy']:.4f}")
        print(f"  Val Prec:   {val_metrics['precision']:.4f}")
        print(f"  Val Rec:    {val_metrics['recall']:.4f}")
        print(f"  Val F1:     {val_metrics['f1']:.4f}")
        if val_metrics.get('cat_accuracy') is not None:
            print(f"  Val Cat Acc:{val_metrics['cat_accuracy']:.4f}")
        print(f"{'='*60}\n")
    
    # Zapis modelu
    print("\n💾 Zapis modelu...")
    torch.save(model.state_dict(), '../weights/auction_model.pt')
    print("✓ Zapisano: weights/auction_model.pt")
    
    # Zapis historii
    with open('../weights/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print("✓ Zapisano: weights/training_history.json")
    
    print("\n✅ Trening ukończony!")

if __name__ == '__main__':
    main()