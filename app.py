from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import io
import torchvision.transforms as transforms

app = FastAPI()

# ✅ อนุญาตให้ React Native หรือเว็บแอปเรียก API นี้ได้
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # คุณสามารถเปลี่ยนเป็น origin ของแอป React Native ได้
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🧠 โหลดโมเดล PyTorch (.pt) ที่แปลงด้วย torch.jit.save แล้ว
model = torch.jit.load("flower_model_mobile.pt", map_location=torch.device("cpu"))
model.eval()

# 📐 Transform ต้องตรงกับตอนเทรน
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# ✅ หน้า root สำหรับเช็คสถานะ API
@app.get("/")
def root():
    return {"message": "🌸 Flower Classifier API is running!"}

# 📷 Endpoint สำหรับรับรูปภาพแล้วพยากรณ์ class
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)  # เพิ่ม batch dim

    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).item()

    return {"class_index": pred}
