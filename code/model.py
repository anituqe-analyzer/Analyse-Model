# model.py
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
from torchvision.models import efficientnet_b0

class AuctionAuthenticityModel(nn.Module):
    def __init__(self, num_classes=3, device='cpu'):  # 3 klasy!
        super().__init__()
        self.device = device
        
        # Vision
        self.vision_model = efficientnet_b0(pretrained=True)
        self.vision_model.classifier = nn.Identity()
        vision_out_dim = 1280
        
        # Text
        self.text_model = DistilBertModel.from_pretrained(
            'distilbert-base-multilingual-cased'
        )
        text_out_dim = 768
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(
            'distilbert-base-multilingual-cased'
        )
        
        # Fusion (bez BatchNorm!)
        hidden_dim = 256
        self.fusion = nn.Sequential(
            nn.Linear(vision_out_dim + text_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, images, texts):
        vision_features = self.vision_model(images)
        tokens = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors='pt'
        ).to(self.device)
        text_outputs = self.text_model(**tokens)
        text_features = text_outputs.last_hidden_state[:, 0, :]
        
        combined = torch.cat([vision_features, text_features], dim=1)
        logits = self.fusion(combined)
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    print("Testowanie modelu...")
    
    device = torch.device('cpu')
    model = AuctionAuthenticityModel(device=device).to(device)
    
    print(f"✓ Model stworzony")
    print(f"  - Parametrów: {model.count_parameters():,}")
    
    # Dummy test
    dummy_img = torch.randn(2, 3, 224, 224).to(device)
    dummy_texts = ["Silver spoon antique", "Polish silverware 19th century"]
    
    with torch.no_grad():
        output = model(dummy_img, dummy_texts)
    
    print(f"✓ Forward pass: {output.shape}")
    print(f"  - Output: {output}")
    
    # Estimate model size
    print(f"\n📊 Rozmiar modelu:")
    torch.save(model.state_dict(), 'temp_model.pt')
    import os
    size_mb = os.path.getsize('temp_model.pt') / (1024*1024)
    print(f"  - {size_mb:.1f} MB")
    os.remove('temp_model.pt')