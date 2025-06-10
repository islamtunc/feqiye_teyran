# Bismillahirrahmanirrahim
# Elhamdulillahi Rabbil Alamin
# Es-selatu ve selamu ala Resulina Muhammedin ve ala alihi ve sahbihi ecmain
# Allah u Teala bizleri doğru yoldan ayırmasın, ilim ve hikmetle donatsın.
# Allah u Ekber Allah u Ekber, La ilahe illallah, Allahu Ekber, Allahu Ekber, ve lillahi'l-hamd



import tensorflow_hub as hub
import numpy as np
from flask import Flask, render_template_string, request
from transformers import AutoTokenizer, AutoModel

class KnowledgeBase:
    def __init__(self, txt_path, embed_model_url):
        self.txt_path = txt_path
        self.embed = hub.load(embed_model_url)
        self.original_knowledge = self._load_knowledge()
        self.knowledge = [self._preprocess_text(line) for line in self.original_knowledge]
        self.embeddings = self.embed(self.knowledge)

    def _preprocess_text(self, text):
        # Küçük harfe çevir, noktalama işaretlerini kaldır, fazla boşlukları temizle
        import re
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _load_knowledge(self):
        # Satır sonu karakterlerini ve gereksiz boşlukları temizle, sadece anlamlı satırları al
        with open(self.txt_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        # Gereksiz sayfa numarası, boş satır, tek rakam/sayı, web adresi, kısa satırları filtrele
        clean_lines = []
        for line in lines:
            if line.isdigit():
                continue
            if line.startswith('www.') or line.startswith('http'):
                continue
            if len(line) < 10:
                continue
            if line.startswith('Published by:') or line.startswith('More information'):
                continue
            if line.startswith('This brochure'):
                continue
            # Satırı ön işlemeden geçir
            clean_lines.append(self._preprocess_text(line))
        return clean_lines

    def query(self, question):
        q = self._preprocess_text(question)
        q_emb = self.embed([q])
        sims = np.inner(q_emb, self.embeddings)[0]
        idx = np.argmax(sims)
        return self.original_knowledge[idx]

app = Flask(__name__)

HTML = '''
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <title>Asistan Sohbet</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f7f7f7; }
        .container { max-width: 600px; margin: 40px auto; background: #fff; padding: 30px; border-radius: 10px; box-shadow: 0 2px 8px #ccc; }
        h2 { text-align: center; }
        .chat-box { min-height: 200px; margin-bottom: 20px; }
        .user { color: #0074d9; }
        .bot { color: #2ecc40; }
        form { display: flex; }
        input[type=text] { flex: 1; padding: 10px; border: 1px solid #ccc; border-radius: 5px; }
        button { padding: 10px 20px; border: none; background: #0074d9; color: #fff; border-radius: 5px; margin-left: 10px; cursor: pointer; }
        button:hover { background: #005fa3; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Asistan Sohbet</h2>
        <div class="chat-box">
            {% for q, a in history %}
                <div><span class="user">Sen:</span> {{ q }}</div>
                <div><span class="bot">Asistan:</span> {{ a }}</div>
            {% endfor %}
        </div>
        <form method="post">
            <input type="text" name="question" autofocus placeholder="Sorunuzu yazın..." required>
            <button type="submit">Gönder</button>
        </form>
    </div>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def chat():
    if 'history' not in request.cookies:
        history = []
    else:
        import json
        history = json.loads(request.cookies.get('history'))
    if request.method == 'POST':
        q = request.form['question']
        a = kb.query(q)
        history.append((q, a))
    resp = render_template_string(HTML, history=history)
    from flask import make_response
    response = make_response(resp)
    import json
    response.set_cookie('history', json.dumps(history))
    return response

if __name__ == '__main__':
    import os
    if not os.path.exists('knowledge.txt'):
        with open('knowledge.txt', 'w', encoding='utf-8') as f:
            f.write('Merhaba! Benim adım Asistan.\n')
    print('TensorFlow Hub modeli yükleniyor...')
    kb = KnowledgeBase('knowledge.txt', 'https://tfhub.dev/google/universal-sentence-encoder/4')
    print('Asistan hazır! Web arayüzü için http://127.0.0.1:5000 adresine gidin.')
    app.run(debug=True)
