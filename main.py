import os
import random
import numpy as np
from datetime import datetime
import sqlite3
import io
import base64
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from flask import Flask, render_template, redirect, url_for, request, flash
from pathlib import Path


#       CPU Optimizations & Reproducibility

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
torch.set_num_threads(8)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

###############################################
#       Few‑Shot Configuration & Classes
###############################################
BASE_DIR             = Path(__file__).resolve().parent
FEWSHOT_TRAIN_DIR    = BASE_DIR / "fewshot_dataset" / "train"
MODEL_WEIGHTS_H5     = str(BASE_DIR / "best_model.h5")
DEVICE               = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMB_DIM              = 384
SIMILARITY_THRESHOLD = 0.6
DIST_THRESHOLD       = (2 - 2 * SIMILARITY_THRESHOLD) ** 0.5

CLASS_LIST = [
    "Apple___Cedar_apple_rust",
    "Blueberry___healthy",
    "Cherry_(including_sour)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___healthy",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Strawberry___healthy",
    "Tomato___Tomato_mosaic_virus"
]
plant_names   = [c.split('___',1)[0] for c in CLASS_LIST]
disease_names = [c.split('___',1)[1] for c in CLASS_LIST]

###############################################
#           Data Preprocessing Transform
###############################################
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

###############################################
#       Few‑Shot Model Definition
###############################################
class EfficientProtoNet(nn.Module):
    def __init__(self, emb_dim=EMB_DIM):
        super().__init__()
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        orig_cls = backbone.classifier
        if isinstance(orig_cls, nn.Sequential) and hasattr(orig_cls[1], 'in_features'):
            in_f = orig_cls[1].in_features
        elif hasattr(orig_cls, 'in_features'):
            in_f = orig_cls.in_features
        else:
            raise RuntimeError("Unable to determine classifier in_features")
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.embed = nn.Sequential(
            nn.Linear(in_f, 768),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(768, emb_dim),
            nn.BatchNorm1d(emb_dim)
        )

    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.embed(features)
        return F.normalize(embeddings, p=2, dim=1)

def load_model_h5(model, filename):
    state = {}
    with h5py.File(filename, 'r') as f:
        grp = f['model'] if 'model' in f else f
        for k, d in grp.items():
            state[k] = torch.tensor(d[()]).to(DEVICE)
    model.load_state_dict(state, strict=False)
    return model

###############################################
#       Initialize Model & Build Prototypes
###############################################
model = EfficientProtoNet(emb_dim=EMB_DIM).to(DEVICE)
model = load_model_h5(model, MODEL_WEIGHTS_H5)
model.eval()

prototypes = []
for cls in CLASS_LIST:
    cls_dir = FEWSHOT_TRAIN_DIR / cls
    imgs = [p for p in cls_dir.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png')]
    embs = []
    for p in imgs:
        img = Image.open(p).convert('RGB')
        x = val_transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            e = model(x)
        embs.append(e[0].cpu())
    prototypes.append(torch.stack(embs).mean(dim=0))
prototypes = torch.stack(prototypes).to(DEVICE)

def predict_image_with_threshold(image, threshold=DIST_THRESHOLD):
    img_tensor = val_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = model(img_tensor)
    dists = torch.cdist(embedding.float(), prototypes.float())
    min_dist, idx = dists.min(dim=1)
    if min_dist.item() > threshold:
        return "Given image is not leaf", "Given image is not leaf"
    cls_idx = idx.item()
    return plant_names[cls_idx], disease_names[cls_idx]

###############################################
#       SQLite DB Setup & Helpers
###############################################
DB_PATH = str(BASE_DIR / "mydatabase.db")

def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS Users (
            Email    TEXT PRIMARY KEY NOT NULL,
            Name     TEXT NOT NULL,
            password TEXT NOT NULL,
            pet      TEXT NOT NULL,
            Date     TEXT NOT NULL
        )
    """)
    con.commit()
    con.close()

init_db()

###############################################
#             Flask App & Routes
###############################################
app = Flask(__name__)
app.secret_key = '1234'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

name = ''

@app.route('/')
def landing():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET','POST'])
def login():
    global name
    error = None
    if request.method=='POST':
        email   = request.form['email']
        password= request.form['password']
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        cur.execute("SELECT Name FROM Users WHERE Email=? AND password=?", (email, password))
        row = cur.fetchone()
        con.close()
        if row:
            name = row[0]
            return redirect(url_for('home'))
        else:
            error = "Invalid Credentials. Please try again."
    return render_template('login.html', error=error)

@app.route('/register', methods=['GET','POST'])
def register():
    error = None
    if request.method=='POST':
        name_reg = request.form['name']
        email    = request.form['email']
        password = request.form['password']
        rpassword= request.form['rpassword']
        pet      = request.form['pet']
        if password!=rpassword:
            error='Passwords do not match!'
        else:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            con = sqlite3.connect(DB_PATH)
            cur = con.cursor()
            cur.execute("SELECT 1 FROM Users WHERE Email=?", (email,))
            if cur.fetchone():
                error = "User already registered!"
            else:
                cur.execute(
                    "INSERT INTO Users(Email, Name, password, pet, Date) VALUES(?,?,?,?,?)",
                    (email, name_reg, password, pet, now)
                )
                con.commit()
                con.close()
                return redirect(url_for('login'))
            con.close()
    return render_template('register.html', error=error)

@app.route('/forgot', methods=['GET','POST'])
def forgot():
    error = None
    if request.method=='POST':
        email = request.form['email']
        pet   = request.form['pet']
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        cur.execute("SELECT password FROM Users WHERE Email=? AND pet=?", (email, pet))
        row = cur.fetchone()
        con.close()
        if row:
            error = "Your password: " + row[0]
        else:
            error = "Invalid information. Please try again."
    return render_template('forgot-password.html', error=error)

@app.route('/home')
def home():
    return render_template('home.html', name=name)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', name=name)

@app.route('/image', methods=['GET','POST'])
def image():
    if request.method=='POST':
        file = request.files.get('doc')
        if not file or file.filename=="":
            flash("No image selected for uploading")
            return redirect(request.url)
        try:
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            encoded = base64.b64encode(img_bytes).decode('utf-8')
        except Exception as e:
            flash("Error processing image: "+str(e))
            return redirect(request.url)

        plant_pred, disease_pred = predict_image_with_threshold(img)
        return render_template(
            'image_test.html',
            name=name,
            result=None if "not leaf" in plant_pred else plant_pred,
            suggestion=disease_pred,
            image_data=encoded
        )
    return render_template('image.html', name=name)

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

if __name__=='__main__':
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=True)
