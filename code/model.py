# model.py
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
from torchvision.models import efficientnet_b0
from config import AUTHENTICITY_CLASSES, CATEGORIES

class AuctionAuthenticityModel(nn.Module):
    def __init__(self, num_classes=None, device='cpu'):
        # If num_classes not specified, use config
        if num_classes is None:
            num_classes = len(AUTHENTICITY_CLASSES)
        # Category classes (separate head)
        num_categories = len(CATEGORIES)
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
        
        # Fusion encoder (shared) -> then two heads (authenticity + category)
        hidden_dim = 256
        self.fusion_encoder = nn.Sequential(
            nn.Linear(vision_out_dim + text_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Heads
        self.auth_head = nn.Linear(128, num_classes)
        self.cat_head = nn.Linear(128, num_categories)

        # store sizes for reference
        self.num_classes = num_classes
        self.num_categories = num_categories
    
    def forward(self, images, texts):
        vision_features = self.vision_model(images)
        tokens = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors='pt'
        ).to(self.device)
        text_outputs = self.text_model(**tokens)
        text_features = text_outputs.last_hidden_state[:, 0, :]

        combined = torch.cat([vision_features, text_features], dim=1)
        shared = self.fusion_encoder(combined)

        auth_logits = self.auth_head(shared)
        cat_logits = self.cat_head(shared)

        # probabilities
        auth_probs = torch.softmax(auth_logits, dim=1)
        cat_probs = torch.softmax(cat_logits, dim=1)

        return {
            'auth_logits': auth_logits,
            'auth_probs': auth_probs,
            'cat_logits': cat_logits,
            'cat_probs': cat_probs,
        }

    def compute_loss(self, outputs, auth_labels=None, cat_labels=None, auth_weight=1.0, cat_weight=1.0):
        """Compute combined loss for two heads. Labels should be LongTensors on same device.

        Returns combined scalar loss and a dict with individual losses.
        """
        losses = {}
        loss = 0.0
        criterion = nn.CrossEntropyLoss()

        if auth_labels is not None:
            l_auth = criterion(outputs['auth_logits'], auth_labels)
            losses['auth_loss'] = l_auth
            loss = loss + auth_weight * l_auth

        if cat_labels is not None:
            # Allow sentinel -1 for unknown/uncertain categories and ignore them
            if cat_labels.dim() == 1:
                mask = cat_labels >= 0
            else:
                mask = (cat_labels.squeeze(-1) >= 0)

            if mask.sum().item() > 0:
                selected_logits = outputs['cat_logits'][mask]
                selected_labels = cat_labels[mask]
                l_cat = criterion(selected_logits, selected_labels)
                losses['cat_loss'] = l_cat
                loss = loss + cat_weight * l_cat
            else:
                # No valid category labels in batch
                losses['cat_loss'] = torch.tensor(0.0, device=self.device)

        return loss, losses
    
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

    # Print shapes
    print("✓ Forward pass:")
    print(f"  - auth_logits: {output['auth_logits'].shape}")
    print(f"  - auth_probs: {output['auth_probs'].shape}")
    print(f"  - cat_logits: {output['cat_logits'].shape}")
    print(f"  - cat_probs: {output['cat_probs'].shape}")

    # Show predicted labels and top probabilities
    auth_pred = torch.argmax(output['auth_probs'], dim=1)
    cat_pred = torch.argmax(output['cat_probs'], dim=1)

    for i in range(output['auth_probs'].shape[0]):
        a_idx = int(auth_pred[i].item())
        a_prob = float(output['auth_probs'][i, a_idx].item())
        c_idx = int(cat_pred[i].item())
        c_prob = float(output['cat_probs'][i, c_idx].item())
        a_name = AUTHENTICITY_CLASSES.get(a_idx, str(a_idx))
        c_name = CATEGORIES.get(c_idx, str(c_idx))
        print(f"\nSample {i}:")
        print(f"  - Authenticity: {a_name} ({a_prob:.3f})")
        print(f"  - Category: {c_name} ({c_prob:.3f})")
    
    # Estimate model size
    print(f"\n📊 Rozmiar modelu:")
    torch.save(model.state_dict(), 'temp_model.pt')
    import os
    size_mb = os.path.getsize('temp_model.pt') / (1024*1024)
    print(f"  - {size_mb:.1f} MB")
    os.remove('temp_model.pt')