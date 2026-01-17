import os
import cv2
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash

# 1. INISIALISASI FLASK 
app = Flask(__name__)
app.secret_key = "satria_secret_id"

# 2. IMPORT AI & FIREBASE
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import firebase_admin
from firebase_admin import credentials, firestore

# 3. KONFIGURASI FIREBASE 
try:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred) 
    db = firestore.client()
    firebase_connected = True
    print("selamat firebase terhubung!")
except Exception as e:
    firebase_connected = False
    print(f" Firebasemu lagi error: {e}. Menjalankan Mode Lokal.")

# 4. LOAD MODEL AI & SETUP FOLDER
print("Tunggu Boy lagi muat model AI MobileNetV2...")
model = MobileNetV2(weights='imagenet')

UPLOAD_FOLDER = 'static/uploads'
FACES_FOLDER = 'database/faces'
for f in [UPLOAD_FOLDER, FACES_FOLDER]:
    if not os.path.exists(f): os.makedirs(f)

# 5. FUNGSI HELPER  
def get_dog_prediction(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return decode_predictions(preds, top=3)[0]

def compare_faces(ref_path, login_path):
    img1 = cv2.imread(ref_path)
    img2 = cv2.imread(login_path)
    if img1 is None or img2 is None: return 0
    
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    h1 = cv2.calcHist([g1], [0], None, [256], [0, 256])
    h2 = cv2.calcHist([g2], [0], None, [256], [0, 256])
    cv2.normalize(h1, h1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(h2, h2, 0, 1, cv2.NORM_MINMAX)
    return cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)

# 6. ROUTES 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['POST'])
def register():
    user = request.form.get('username', '').strip()
    pw = request.form.get('password')
    face = request.files.get('face_image')
    
    if not user or not pw:
        return jsonify({"status": "error", "message": "Username/Password Kosong!"})

    hashed_pw = generate_password_hash(pw)
    face_path = os.path.join(FACES_FOLDER, f"{user}.jpg")
    if face: face.save(face_path)

    if firebase_connected:
        db.collection('users').document(user).set({
            "password": hashed_pw,
            "face_path": face_path,
            "created_at": datetime.now()
        })
    return jsonify({"status": "success", "message": f"User {user} Berhasil Terdaftar!"})

@app.route('/login-face', methods=['POST'])
def login_face():
    user = request.form.get('username', '').strip()
    face_file = request.files.get('face_image')
    
    # Cek di folder lokal (Fallback)
    ref_path = os.path.join(FACES_FOLDER, f"{user}.jpg")
    if not os.path.exists(ref_path):
        return jsonify({"status": "error", "message": "Wajah belum terdaftar!"})

    temp_path = os.path.join(UPLOAD_FOLDER, "temp_login.jpg")
    face_file.save(temp_path)
    score = compare_faces(ref_path, temp_path)
    
    if score > 0.50: # Toleransi histogram
        return jsonify({"status": "success", "message": "Login Berhasil!", "score": round(score, 2)})
    return jsonify({"status": "fail", "message": "Wajah tidak cocok!", "score": round(score, 2)})

@app.route('/login-password', methods=['POST'])
def login_password():
    user = request.form.get('username', '').strip()
    pw = request.form.get('password')

    if not user or not pw:
        return jsonify({"status": "error", "message": "Isi username & password!"})

    if firebase_connected:
        # 1. Cari user di Firestore
        user_ref = db.collection('users').document(user).get()
        if not user_ref.exists:
            return jsonify({"status": "error", "message": "User tidak ditemukan!"})

        user_data = user_ref.to_dict()
        
        # 2. Verifikasi Password (Hash vs Plain Text)
        if check_password_hash(user_data['password'], pw):
            return jsonify({"status": "success", "message": f"Welcome back, {user}!"})
        else:
            return jsonify({"status": "fail", "message": "Password salah!"})
            
    return jsonify({"status": "error", "message": "Database tidak terhubung!"})

@app.route('/predict', methods=['POST'])
def predict():
    user = request.form.get('username', 'Guest')
    file = request.files.get('file')
    if not file: return jsonify({"status": "error", "message": "Pilih file!"})
    
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
    local_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(local_path)
    
    # 1. Prediksi Top 3
    preds = get_dog_prediction(local_path)
    top_3 = [{"breed": b.replace('_',' ').title(), "conf": f"{p*100:.2f}%"} for (i, b, p) in preds]
    
    # 2. Path Gambar untuk UI (Lokal karena Storage Off)
    img_url = f"/static/uploads/{filename}"

    # 3. Simpan Riwayat ke Firestore Cloud
    if firebase_connected:
        try:
            db.collection('history').add({
                "username": user,
                "predictions": top_3,
                "image_url": img_url,
                "timestamp": datetime.now()
            })
        except Exception as e:
            print(f"Firestore Error: {e}")

    return jsonify({"status": "success", "predictions": top_3, "url": img_url})

@app.route('/get-history', methods=['GET'])
def get_history():
    if not firebase_connected: return jsonify([])
    try:
        docs = db.collection('history').order_by('timestamp', direction=firestore.Query.DESCENDING).limit(5).stream()
        history = [doc.to_dict() for doc in docs]
        return jsonify(history)
    except:
        return jsonify([])

if __name__ == '__main__':
    app.run(debug=True, port=5000) 