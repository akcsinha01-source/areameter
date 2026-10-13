from fastapi import FastAPI, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import base64
import traceback

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/calculate_area")
async def calculate_area(
    scale_ratio: float = Form(...), unit: str = Form(...),
    click_x: int = Form(...), click_y: int = Form(...),
    image: UploadFile = File(...)
):
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return {"success": False, "error": "इमेज सही से नहीं खुल पाई।"}
            
        # पुराना और सटीक थ्रेशोल्ड (जो बिल्कुल सही एरिया निकाल रहा था)
        _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY) 
        
        h, w = thresh.shape
        mask = np.zeros((h + 2, w + 2), np.uint8)
        
        if click_x < 0 or click_y < 0 or click_x >= w or click_y >= h:
            return {"success": False, "error": "क्लिक मैप के बाहर हुआ है।"}
            
        if thresh[click_y, click_x] == 0:
            return {"success": False, "error": "आपने प्लॉट की बाउंड्री (काली लाइन) पर क्लिक किया है। कृपया सफेद जगह पर क्लिक करें।"}
            
        # 1. बिल्कुल सटीक एरिया कैलकुलेशन
        cv2.floodFill(thresh, mask, (click_x, click_y), 100)
        pixel_area = np.sum(thresh == 100)
        
        if pixel_area == 0:
            return {"success": False, "error": "कोई एरिया नहीं मिला।"}
            
        real_area = pixel_area * (scale_ratio ** 2)
        
        # 2. स्मार्ट और फास्ट हाईलाइट (सिर्फ प्लॉट को काटकर भेजना)
        filled_mask = np.zeros_like(thresh)
        filled_mask[thresh == 100] = 255
        
        x, y, bw, bh = cv2.boundingRect(filled_mask)
        
        cropped_mask = filled_mask[y:y+bh, x:x+bw]
        colored_crop = np.zeros((bh, bw, 4), dtype=np.uint8)
        colored_crop[cropped_mask == 255] = [0, 255, 0, 255] # गाढ़ा हरा रंग
        
        _, buffer = cv2.imencode('.png', colored_crop)
        highlighted_b64 = "data:image/png;base64," + base64.b64encode(buffer).decode('utf-8')
        
        return {
            "success": True, 
            "pixel_area": float(pixel_area), 
            "real_area": round(real_area, 2), 
            "unit": unit,
            "highlighted_image": highlighted_b64,
            "box_x": int(x),
            "box_y": int(y),
            "box_w": int(bw),
            "box_h": int(bh)
        }
    except Exception as e:
        traceback.print_exc()
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    print("सर्वर चालू! सटीक एरिया कैलकुलेशन और हाईलाइट सिस्टम एक्टिव...")
    uvicorn.run(app, host="127.0.0.1", port=8000)