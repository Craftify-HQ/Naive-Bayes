# Naive Bayes Review Classifier

โปรเจกต์ทำนายคะแนนรีวิว (rating) จากข้อความรีวิวด้วย Naive Bayes

## ขั้นตอน
1. โหลดข้อมูล (`load_data.py`)
2. ทำความสะอาด/ตัดคำ (`preprocess.py`)
3. สร้างฟีเจอร์ BoW / TF-IDF (`features.py`)
4. เทรนโมเดล (`train.py`)
5. ประเมินผล (`evaluate.py`)

## วิธีรัน
```bash
pip install -r requirements.txt
python train.py
python evaluate.py
```

## ข้อมูล
`w_review_train.csv` : คอลัมน์ `review` (ข้อความ) และ `rating` (คะแนน)

## ผลลัพธ์
(เติมหลังรัน evaluate.py)
