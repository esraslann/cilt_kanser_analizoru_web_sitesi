
from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Modeli yükleme
MODEL_PATH = "mobilNet_model.h5"  # Model dosyasının yolu
model = load_model(MODEL_PATH)  # Modeli yükle

# Sınıf etiketleri
labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

# Yüklenen dosyaların kaydedileceği klasör
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Assets klasörüne erişim için bir route ekliyoruz
@app.route('/assets/<path:filename>')
def assets(filename):
    return send_from_directory('assets', filename)

@app.route('/static/uploads/<path:filename>')
def uploaded_file(filename):
    """Yüklenmiş görselleri sunar."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Anasayfa route'u
@app.route('/')
def index():
    #Index sayfasını render eder.
    return render_template('index.html')

# Kanser türü açıklamaları
cancer_info = {
    'akiec': 'AKIEC, ciltte görülen ve genellikle kansere dönüşme potansiyeli olan keratoz türlerindendir.',
    'bcc': 'BCC (Bazal Hücreli Karsinom), ciltte en sık görülen kanser türüdür ve genellikle yavaş ilerler.',
    'bkl': 'BKL, iyi huylu keratoz lezyonlarıdır ve kansere dönüşme riski yoktur.',
    'df': 'DF (Dermatofibrom), ciltteki bağ dokusundan kaynaklanan iyi huylu bir tümördür.',
    'mel': 'Melanom, cilt kanserinin en ölümcül türüdür ve genellikle pigment üreten hücrelerden kaynaklanır.',
    'nv': 'NV (Nevüs), genellikle iyi huylu cilt benleridir ancak bazı durumlarda kansere dönüşebilir.',
    'vasc': 'Vasküler lezyonlar, damar yapılarından kaynaklanan iyi veya kötü huylu cilt değişimleridir.'
}

# Tahmin API'si
@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files: 
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Dosyayı kaydetme
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    try:
        # İşlem süresini ölçmek için başlangıç zamanı
        import time
        start_time = time.time()

        # Görseli işleme
        img = cv2.imread(file_path)
        img_resized = cv2.resize(img, (224, 224))
        img_array = np.array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Tahmin yapma
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = predictions[0][predicted_class] * 100
        class_name = labels[predicted_class]

        # Tahmin süresini ölç ve yazdır
        print(f"Tahmin süresi: {time.time() - start_time} saniye")

        # Güven oranı %70'in altındaysa
        if confidence < 70:
            return jsonify({
                'message': 'Cilt kanseri riski taşımıyorsunuz.',
                'file_path': f"/static/uploads/{os.path.basename(file_path)}"
            })

        # Güven oranı %70'in üstünde ise
        return jsonify({
            'label': class_name,
            'confidence': f"{confidence:.2f}%",
            'info': cancer_info.get(class_name, 'Bu kanser türü hakkında bilgi mevcut değil.'),
            'file_path': f"/static/uploads/{os.path.basename(file_path)}"
        })

    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500



if __name__ == '__main__':
    app.run(debug=True)