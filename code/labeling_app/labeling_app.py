from flask import Flask, render_template, request, jsonify, send_file
import json
import os
from pathlib import Path

app = Flask(__name__)

# WAŻNE: ustaw ścieżkę POPRAWNIE (zależy gdzie masz folder)
DATASET_PATH = Path(__file__).parent.parent.parent / 'dataset' / 'dataset.json'
RAW_DATA_PATH = Path(__file__).parent.parent.parent / 'dataset' / 'raw_data'

print(f"Dataset path: {DATASET_PATH}")
print(f"Raw data path: {RAW_DATA_PATH}")

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
        if auction.get('label_confidence', 0) == 0:
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
                'current': i + 1
            })
    
    return jsonify({'error': 'Wszystkie aukcje etykietowane!'})

@app.route('/api/save_label', methods=['POST'])
def save_label():
    data = request.json
    dataset = load_dataset()
    
    auction_index = data['auction_index']
    dataset[auction_index]['label'] = data['label']
    dataset[auction_index]['label_confidence'] = data['confidence']
    
    save_dataset(dataset)
    return jsonify({'status': 'ok'})

@app.route('/api/stats')
def get_stats():
    dataset = load_dataset()
    
    total = len(dataset)
    labeled = len([a for a in dataset if a.get('label_confidence', 0) > 0])
    unlabeled = total - labeled
    
    by_label = {
        'ORIGINAL': len([a for a in dataset if a.get('label') == 0]),
        'SCAM': len([a for a in dataset if a.get('label') == 1]),
        'REPLICA': len([a for a in dataset if a.get('label') == 2])
    }
    
    return jsonify({
        'total': total,
        'labeled': labeled,
        'unlabeled': unlabeled,
        'by_label': by_label,
        'progress': round(labeled / total * 100, 1) if total > 0 else 0
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
