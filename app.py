from flask import Flask, request, jsonify, render_template
from infinite_craft_arithmetic_fixed import InfiniteCraftArithmetic
from word_to_emoji import best_emoji

app = Flask(__name__)
craft = InfiniteCraftArithmetic()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/emoji', methods=['POST'])
def get_emoji():
    data = request.get_json(force=True)
    concept = data.get('concept')
    if not concept:
        return jsonify({'error': 'missing concept'}), 400
    emoji, name, method = best_emoji(concept)
    return jsonify({'emoji': emoji, 'name': name, 'method': method})

@app.route('/combine', methods=['POST'])
def combine():
    data = request.get_json(force=True)
    a = data.get('a')
    b = data.get('b')
    if not a or not b:
        return jsonify({'error': 'missing concepts'}), 400
    try:
        result = craft.combine_enhanced(a, b, k=1)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    concept = result[0][0] if result else 'unknown'
    emoji, name, method = best_emoji(concept)  # Get all three values
    print(f"DEBUG Flask: {concept} → {emoji} ({name}) [{method}]")  # Add this debug line
    return jsonify({'result': concept, 'emoji': emoji})

if __name__ == '__main__':
    app.run(debug=True, port=8001)
