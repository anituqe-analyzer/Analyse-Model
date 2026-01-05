import os
import json
from pathlib import Path
from typing import Dict, List

def parse_info_txt(info_path: str) -> Dict:
    """
    Parsuje info.txt z aukcji
    """
    with open(info_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    metadata = {}
    
    # TITLE
    if 'TITLE:' in content:
        title_start = content.find('TITLE:') + len('TITLE:')
        title_end = content.find('\n', title_start)
        metadata['title'] = content[title_start:title_end].strip()
    else:
        metadata['title'] = 'Unknown'
    
    # LINK
    if 'LINK:' in content:
        link_start = content.find('LINK:') + len('LINK:')
        link_end = content.find('\n', link_start)
        metadata['link'] = content[link_start:link_end].strip()
    else:
        metadata['link'] = ''
    
    # PARAMETERS
    metadata['parameters'] = {}
    if 'PARAMETERS:' in content:
        params_start = content.find('PARAMETERS:') + len('PARAMETERS:')
        params_end = content.find('----', params_start)
        if params_end == -1:
            params_end = content.find('DESCRIPTION:', params_start)
        
        params_text = content[params_start:params_end]
        
        for line in params_text.split('\n'):
            if line.strip().startswith('*'):
                line_clean = line.strip()[2:]
                if ':' in line_clean:
                    key, value = line_clean.split(':', 1)
                    metadata['parameters'][key.strip()] = value.strip()
    
    # DESCRIPTION
    if 'DESCRIPTION:' in content:
        desc_start = content.find('DESCRIPTION:') + len('DESCRIPTION:')
        metadata['description'] = content[desc_start:].strip()
    else:
        metadata['description'] = ''
    
    return metadata

def organize_dataset(root_dir: str, output_json: str = 'dataset/dataset.json'):
    """
    Skanuje strukturę i tworzy dataset.json
    """
    root = Path(root_dir)
    dataset = []
    
    for platform_dir in sorted(root.iterdir()):
        if not platform_dir.is_dir():
            continue
        
        platform_name = platform_dir.name
        print(f"\n📁 Platform: {platform_name}")
        
        for auction_dir in sorted(platform_dir.iterdir()):
            if not auction_dir.is_dir():
                continue
            
            auction_id = auction_dir.name
            info_txt = auction_dir / 'info.txt'
            
            if not info_txt.exists():
                print(f"  ⚠️  {auction_id} - brak info.txt")
                continue
            
            try:
                metadata = parse_info_txt(str(info_txt))
            except Exception as e:
                print(f"  ❌ {auction_id} - błąd: {e}")
                continue
            
            # Zbierz zdjęcia
            images = sorted([
                img.name for img in auction_dir.glob('*.jpg')
            ])
            images += sorted([
                img.name for img in auction_dir.glob('*.png')
            ])
            
            if not images:
                print(f"  ⚠️  {auction_id} - brak zdjęć")
                continue
            
            entry = {
                'id': f"{platform_name}_{auction_id}",
                'platform': platform_name,
                'folder_path': str(auction_dir.relative_to(root)),
                'image_count': len(images),
                'images': images,
                'label': 0,  # Default: authentic
                'label_confidence': 0.0,  # Do ręcznego wypełnienia
                **metadata
            }
            
            dataset.append(entry)
            print(f"  ✓ {auction_id} ({len(images)} zdjęć)")
    
    # Zapis
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Dataset wczytany: {len(dataset)} aukcji")
    print(f"💾 Zapisano: {output_json}")
    
    return dataset

if __name__ == '__main__':
    dataset = organize_dataset('dataset/raw_data')
    
    if dataset:
        print("\n" + "="*60)
        print("PRZYKŁAD PIERWSZEJ AUKCJI:")
        print("="*60)
        print(json.dumps(dataset[0], indent=2, ensure_ascii=False)[:800])