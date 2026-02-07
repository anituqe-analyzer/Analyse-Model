from flask import Flask, render_template, request, jsonify, send_file
import json
import os
from pathlib import Path
import sys

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import AUTHENTICITY_CLASSES, CATEGORIES, UNCERTAIN_CATEGORY

app = Flask(__name__)

# WAŻNE: ustaw ścieżkę POPRAWNIE (zależy gdzie masz folder)
DATASET_PATH = Path(__file__).parent.parent.parent / 'dataset' / 'dataset.json'
RAW_DATA_PATH = Path(__file__).parent.parent.parent / 'dataset' / 'raw_data'

print(f"Dataset path: {DATASET_PATH}")
print(f"Raw data path: {RAW_DATA_PATH}")
print(f"Authenticity classes: {AUTHENTICITY_CLASSES}")
print(f"Categories: {CATEGORIES}")

def load_dataset():
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_dataset(data):
    with open(DATASET_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

@app.route('/')
def index():
    dataset = load_dataset()
    return render_template('labeling.html', total_auctions=len(dataset))

@app.route('/image/<path:image_path>')
def serve_image(image_path):
    """Serwuj zdjęcie"""
    full_path = RAW_DATA_PATH / image_path
    print(f"Szukam: {full_path}")
    if full_path.exists():
        return send_file(full_path)
    return "Not found", 404

@app.route('/api/next_unlabeled')
def next_unlabeled():
    dataset = load_dataset()
    
    for i, auction in enumerate(dataset):
        # Check if both authenticity and category are unlabeled
        if auction.get('label_confidence', 0) == 0 or auction.get('category_confidence', 0) == 0:
            # Przygotuj WSZYSTKIE zdjęcia
            images = []
            for img_name in auction['images']:
                img_path = f"{auction['folder_path']}/{img_name}"
                images.append(f"/image/{img_path}")
            
            return jsonify({
                'index': i,
                'id': auction['id'],
                'title': auction['title'],
                'description': auction['description'][:300] + '...',
                'platform': auction['platform'],
                'link': auction['link'],
                'parameters': auction.get('parameters', {}),
                'images': images,
                'total': len(dataset),
                'current': i + 1,
                'authenticity_labels': AUTHENTICITY_CLASSES,
                'categories': CATEGORIES,
                'current_authenticity': auction.get('label'),
                'current_category': auction.get('category')
            })
    
    return jsonify({'error': 'Wszystkie aukcje etykietowane!'})

@app.route('/api/save_label', methods=['POST'])
def save_label():
    data = request.json
    dataset = load_dataset()
    
    auction_index = data['auction_index']
    
    # Save authenticity label
    if 'label' in data:
        dataset[auction_index]['label'] = data['label']
        dataset[auction_index]['label_confidence'] = data.get('confidence', 1)
    
    # Save category label
    if 'category' in data:
        dataset[auction_index]['category'] = data['category']
        dataset[auction_index]['category_confidence'] = data.get('category_confidence', 1)
    
    save_dataset(dataset)
    return jsonify({'status': 'ok'})

@app.route('/api/stats')
def get_stats():
    dataset = load_dataset()
    
    total = len(dataset)
    labeled_auth = len([a for a in dataset if a.get('label_confidence', 0) > 0])
    unlabeled_auth = total - labeled_auth
    
    labeled_cat = len([a for a in dataset if a.get('category_confidence', 0) > 0])
    unlabeled_cat = total - labeled_cat
    
    by_authenticity = {
        AUTHENTICITY_CLASSES[i]: len([a for a in dataset if a.get('label') == i])
        for i in range(len(AUTHENTICITY_CLASSES))
    }
    
    by_category = {
        CATEGORIES[i]: len([a for a in dataset if a.get('category') == i])
        for i in range(len(CATEGORIES))
    }
    
    # Add Uncertain count
    uncertain_count = len([a for a in dataset if a.get('category') == 'uncertain'])
    if uncertain_count > 0:
        by_category[UNCERTAIN_CATEGORY] = uncertain_count
    
    return jsonify({
        'total': total,
        'authenticity': {
            'labeled': labeled_auth,
            'unlabeled': unlabeled_auth,
            'by_label': by_authenticity,
            'progress': round(labeled_auth / total * 100, 1) if total > 0 else 0
        },
        'category': {
            'labeled': labeled_cat,
            'unlabeled': unlabeled_cat,
            'by_label': by_category,
            'progress': round(labeled_cat / total * 100, 1) if total > 0 else 0
        }
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
