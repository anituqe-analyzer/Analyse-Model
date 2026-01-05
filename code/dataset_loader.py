import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class AuctionDatasetFromJSON(Dataset):
    def __init__(self, json_path: str, root_dir: str, transform=None, max_samples=None):
        """
        json_path: dataset/dataset.json
        root_dir: dataset/raw_data
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        if max_samples:
            self.data = self.data[:max_samples]
        
        self.root_dir = Path(root_dir)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        auction = self.data[idx]
        
        # Ścieżka do zdjęcia
        img_path = self.root_dir / auction['folder_path'] / auction['images'][0]
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Błąd wczytywania {img_path}: {e}")
            # Fallback: czarne zdjęcie
            img = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            img = self.transform(img)
        
        # Tekst: title + opis
        text = f"{auction.get('title', '')} {auction.get('description', '')}"
        
        return {
            'image': img,
            'text': text,
            'platform': auction['platform'],
            'title': auction['title'],
            'id': auction['id'],
            'label': torch.tensor(auction.get('label', 0), dtype=torch.long),
            'folder_path': auction['folder_path']
        }

# Transformacje
get_transforms = lambda: transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

if __name__ == '__main__':
    print("Testowanie DataLoadera...")
    
    dataset = AuctionDatasetFromJSON(
        json_path='../dataset/dataset.json',
        root_dir='../dataset/raw_data',
        transform=get_transforms(),
        max_samples=5
    )
    
    print(f"✓ Dataset załadowany: {len(dataset)} próbek")
    
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    
    for batch in loader:
        print(f"\nBatch:")
        print(f"  - Image shape: {batch['image'].shape}")
        print(f"  - Texts: {len(batch['text'])}")
        print(f"  - Platforms: {batch['platform']}")
        print(f"  - Labels: {batch['label']}")
        print(f"  - Example text: {batch['text'][0][:100]}...")
        break