import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from datetime import datetime

# 1. ตั้งค่าหน้าเว็บให้เป็นแบบ Wide
st.set_page_config(
    page_title="Smart Rice Classification System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. ตกแต่ง CSS และกำหนดกรอบล็อกรูปภาพให้อยู่ภายในกรอบเรียบร้อย
st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; font-weight: 800; color: #1E3A8A; margin-bottom: 0px; }
    .sub-header { font-size: 1.05rem; color: #4B5563; margin-bottom: 15px; }
    
    .metric-card {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        padding: 22px; border-radius: 16px; border: 1px solid #A7F3D0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); text-align: center; margin-bottom: 20px;
    }
    .recipe-box {
        background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
        padding: 18px; border-radius: 16px;
        border: 1px solid #FDE68A; margin-top: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.03);
    }
    .info-box { 
        background-color: #F8FAFC; padding: 18px; border-radius: 16px; 
        border: 1px solid #E2E8F0; text-align: center; color: #64748B;
    }
    </style>
""", unsafe_allow_html=True)

# 3. โหลดโมเดล AI
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('rice_model.keras')

with st.spinner('กำลังโหลดโมเดล AI เข้าสู่ระบบ...'):
    model = load_model()

class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']
IMG_SIZE = (128, 128)

# ฐานข้อมูลเมนูอาหารแนะนำ
recipe_db = {
    'Arborio': {
        'dish': 'รีซอตโต้ครีมเห็ดทรัฟเฟิล (Mushroom Risotto)',
        'desc': 'ข้าวเมล็ดสั้น มีแป้งสูง ให้เนื้อสัมผัสเข้มข้น หอมมัน และหนุบหนับ (Al dente)'
    },
    'Basmati': {
        'dish': 'ข้าวหมกบริยานีไก่ (Chicken Biryani)',
        'desc': 'ข้าวเมล็ดเรียวยาว หอมกลิ่นเครื่องเทศเด่นชัด หุงสุกแล้วร่วนซุย ไม่เกาะก้อน'
    },
    'Ipsala': {
        'dish': 'ข้าวอบสไตล์ดั้งเดิม (Turkish Baldo Pilaf)',
        'desc': 'ข้าวพันธุ์ท้องถิ่นจากตุรกี เมล็ดใหญ่หนา ดูดซับน้ำซุปได้ดีเยี่ยม'
    },
    'Jasmine': {
        'dish': 'ข้าวผัดปู หรือ ทานคู่แกงเผ็ดไทย',
        'desc': 'ข้าวหอมมะลิไทย นุ่มเหนียว มีกลิ่นหอมดอกไม้อ่อนๆ เข้ากับกับข้าวไทย'
    },
    'Karacadag': {
        'dish': 'ข้าวกล้องภูเขาไฟอบสมุนไพร (Anatolian Mountain Rice)',
        'desc': 'ข้าวปลูกบนพื้นที่ภูเขาไฟ ให้โปรตีนสูง เมล็ดคงรูปได้ดีเยี่ยม'
    }
}

# ระบบประวัติการวิเคราะห์
if 'history' not in st.session_state:
    st.session_state.history = []

# Sidebar ด้านข้าง
with st.sidebar:
    st.markdown("### 🌾 เกี่ยวกับโปรเจกต์")
    st.markdown("ระบบจำแนกสายพันธุ์ข้าวด้วย **MobileNetV2** + **Center-Crop Pipeline**")
    st.divider()
    
    st.markdown("### 📋 ประเภทข้าวที่รองรับ:")
    for name in class_names:
        st.markdown(f"- **{name}**")
    st.divider()
    
    st.markdown("### 🕒 ประวัติการวิเคราะห์")
    if st.session_state.history:
        for item in reversed(st.session_state.history[-5:]):
            st.markdown(f"**{item['time']}**<br>🎯 `{item['class']}` ({item['conf']:.1f}%)", unsafe_allow_html=True)
            st.divider()
        if st.button("ล้างประวัติทั้งหมด", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    else:
        st.caption("ยังไม่มีประวัติการทดสอบ")

# หน้าจอหลัก
st.markdown('<p class="main-header">🌾 Smart Rice Classification System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">ระบบปัญญาประดิษฐ์วิเคราะห์และจำแนกสายพันธุ์เมล็ดข้าว พร้อมแนะนำเมนูอาหาร</p>', unsafe_allow_html=True)
st.divider()

col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.markdown("#### 📥 อัปโหลดรูปภาพเมล็ดข้าว")
    uploaded_file = st.file_uploader("เลือกไฟล์ภาพ (รองรับ JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        # ควบคุมขนาดภาพไม่ให้หลุดกรอบด้วยพารามิเตอร์ width แบบระบุค่าพิกเซลตายตัว
        st.image(image, caption='ภาพถ่ายที่นำเข้าสู่ระบบ', width=380)
    else:
        st.markdown("""
            <div style="background-color: #F1F5F9; padding: 40px; border-radius: 16px; border: 1px solid #CBD5E1; text-align: center; color: #94A3B8;">
                <p style="margin: 0; font-size: 1.1rem;">📷 ยังไม่มีรูปภาพที่อัปโหลด</p>
                <p style="margin: 5px 0 0 0; font-size: 0.85rem;">(กรุณาเลือกไฟล์ภาพด้านบน)</p>
            </div>
        """, unsafe_allow_html=True)

with col_right:
    st.markdown("#### 📊 ผลการวิเคราะห์จาก AI")
    
    if uploaded_file is not None:
        analyze_btn = st.button('🚀 เริ่มวิเคราะห์สายพันธุ์ข้าว', type="primary", use_container_width=True)
        
        if analyze_btn:
            with st.spinner('กำลังประมวลผลภาพด้วย AI...'):
                img_np = np.array(image)
                if len(img_np.shape) == 2:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
                elif img_np.shape[2] == 4:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
                    
                h, w, _ = img_np.shape
                crop_scale = 0.7 
                ch, cw = int(h * crop_scale), int(w * crop_scale)
                y1, x1 = (h - ch) // 2, (w - cw) // 2
                cropped_rice = img_np[y1:y1+ch, x1:x1+cw]
                
                resized_rice = cv2.resize(cropped_rice, IMG_SIZE)
                rice_array = np.array(resized_rice, dtype=np.float32)
                rice_array = np.expand_dims(rice_array, axis=0)
                
                preds = model.predict(rice_array, verbose=0)
                pred_idx = np.argmax(preds[0])
                pred_class = class_names[pred_idx]
                confidence = np.max(preds[0]) * 100
                
                # บันทึกลงประวัติ
                current_time = datetime.now().strftime("%H:%M:%S")
                st.session_state.history.append({
                    'time': current_time,
                    'class': pred_class,
                    'conf': confidence
                })
                
                # แสดงผลลัพธ์การทำนาย
                st.markdown(f"""
                    <div class="metric-card">
                        <h2 style="color: #065F46; margin: 0 0 5px 0;">🎯 สายพันธุ์: {pred_class}</h2>
                        <p style="font-size: 1.15rem; color: #047857; margin: 0;">ระดับความมั่นใจ: <b>{confidence:.2f}%</b></p>
                    </div>
                """, unsafe_allow_html=True)
                
                # ส่วนแนะนำเมนูอาหาร
                recipe = recipe_db.get(pred_class, {})
                st.markdown(f"""
                    <div class="recipe-box">
                        <h4 style="color: #92400E; margin-top:0;">🍳 เมนูอาหารแนะนำสำหรับข้าวพันธุ์นี้</h4>
                        <p style="margin-bottom: 5px; color: #78350F;"><b>เมนู:</b> {recipe.get('dish', '-')}</p>
                        <p style="margin-bottom: 0; font-size: 0.95rem; color: #78350F;"><b>ลักษณะเด่น:</b> {recipe.get('desc', '-')}</p>
                    </div>
                """, unsafe_allow_html=True)
                
                # แสดงกราฟความมั่นใจรายคลาส
                st.markdown("##### 📈 คะแนนความมั่นใจแต่ละสายพันธุ์:")
                for i, c_name in enumerate(class_names):
                    score = float(preds[0][i])
                    col_name, col_bar, col_val = st.columns([1.5, 3, 1])
                    with col_name: st.write(f"**{c_name}**")
                    with col_bar: st.progress(score)
                    with col_val: st.write(f"{score * 100:.1f}%")
        else:
            st.markdown("""
                <div class="info-box" style="margin-top: 10px;">
                    <p style="margin: 0; color: #475569;">👉 อัปโหลดภาพเรียบร้อยแล้ว กรุณากดปุ่ม <b>"เริ่มวิเคราะห์สายพันธุ์ข้าว"</b> ด้านบนเพื่อดูผลลัพธ์</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div class="info-box">
                <p style="margin: 0; color: #64748B;">ยังไม่มีข้อมูลการวิเคราะห์ (โปรดอัปโหลดภาพทางฝั่งซ้าย)</p>
            </div>
        """, unsafe_allow_html=True)
        
        # กล่องแสดงประเภทข้าวที่รองรับ
        st.markdown("""
            <div style="background-color: #F8FAFC; padding: 20px; border-radius: 16px; border: 1px solid #E2E8F0; margin-top: 15px;">
                <h4 style="color: #1E3A8A; margin-top:0;">🌾 ข้าว 5 สายพันธุ์ที่ระบบรองรับ:</h4>
                <ul style="color: #4B5563; margin-bottom:0; padding-left: 20px; line-height: 1.6;">
                    <li><b>Arborio</b> (ข้าวอิตาลี เมล็ดอ้วนป้อม เหมาะทำรีซอตโต้)</li>
                    <li><b>Basmati</b> (ข้าวเอเชียใต้ เมล็ดเรียวยาว หอมร่วน)</li>
                    <li><b>Ipsala</b> (ข้าวตุรกี เมล็ดใหญ่หนา เหมาะทำข้าวอบ)</li>
                    <li><b>Jasmine</b> (ข้าวหอมมะลิไทย นุ่มเหนียว มีกลิ่นหอม)</li>
                    <li><b>Karacadag</b> (ข้าวภูเขาไฟตุรกี โปรตีนสูง คงรูปดี)</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)