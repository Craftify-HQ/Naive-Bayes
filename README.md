# Naive-Bayes

จำแนกคะแนนรีวิวร้านอาหารภาษาไทย (1–5 ดาว) ด้วย Multinomial Naive Bayes ที่เขียนเองด้วย NumPy
ใช้ข้อมูลจาก [Wongnai Corpus](https://github.com/wongnai/wongnai-corpus)

## ติดตั้ง
```bash
pip install numpy matplotlib seaborn pythainlp wordcloud scikit-learn
```

## วิธีใช้งาน
```bash
python main.py
```
- ครั้งแรกโปรแกรมจะดาวน์โหลดชุดข้อมูลมาเก็บไว้ที่ `data/w_review_train.csv` ให้อัตโนมัติ (ต้องต่ออินเทอร์เน็ต)
- ครั้งต่อไปจะใช้ไฟล์ที่ดาวน์โหลดไว้แล้ว

## ผลลัพธ์ที่ได้
- ค่าความแม่นยำ (accuracy) บนชุดทดสอบ
- Heatmap แสดงคำเด่นของรีวิวแต่ละระดับดาว
- Word cloud ของคำที่พบบ่อย

## หลักการทำงาน
1. ตัดคำภาษาไทยด้วย PyThaiNLP (engine `newmm`) และตัดคำที่ไม่มีความหมาย (stopwords)
2. แบ่งข้อมูล train/test เป็น 80:20
3. นับความถี่คำในแต่ละระดับดาว แล้วทำ Laplace smoothing (+1)
4. ทายระดับดาวจาก log prior + ผลรวม log likelihood ของคำในรีวิว

## หมายเหตุ
โค้ดตั้งฟอนต์ไว้ที่ `/Library/Fonts/Arial Unicode.ttf` (macOS)
ถ้ารันบน Windows ให้เปลี่ยนค่า `FONT` เป็น `C:/Windows/Fonts/tahoma.ttf` เพื่อให้แสดงภาษาไทยในกราฟได้
