import requests
import csv
import random
import pandas as pd
import os
from dotenv import load_dotenv

# ตั้งค่า API
load_dotenv()
API_KEY = os.getenv("DeepseekAPIkey") 
API_URL = "https://api.deepseek.com/v1/chat/completions"  


headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# โหลดไฟล์ CSV
file_path = "dataset/data.csv"
df = pd.read_csv(file_path)

# สร้างลิสต์เพื่อเก็บคำแรกของคำถามที่ถามไปแล้ว
previous_first_words = []  

# ตั้งค่าไฟล์ CSV สำหรับบันทึกคำถาม
csv_file = "pet_questions_v2.csv"
with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Question"])  # เขียนหัวข้อคอลัมน์

    # ลูป 1000 ครั้ง
    for i in range(1020):
        # สุ่มเลือกแถวจาก dataset
        row = df.sample(n=1).iloc[0]
        animal = row["AnimalName"]
        symptoms = row[1:-1].dropna().tolist()  # ดึงเฉพาะคอลัมน์อาการ (ไม่นับ AnimalName และ Dangerous)

        # สุ่มจำนวนอาการ 1-3 อย่างสุ่ม (ถ้ามีมากพอ)
        num_symptoms = random.randint(1, min(3, len(symptoms)))
        selected_symptoms = random.sample(symptoms, num_symptoms) if symptoms else ["unknown symptom"]
        symptoms_text = " and ".join(selected_symptoms)

        # ตั้งค่า payload สำหรับคำถาม โดยเพิ่มเงื่อนไขว่าไม่ให้คำถามมีคำแรกที่ซ้ำ
        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": f"I am a pet owner whose pet is currently sick, and I need advice on its symptoms. Generate diverse and unique questions rather than repeating the same structure. Do not ask questions starting with these words: {', '.join(previous_first_words)}."},
                {"role": "user", "content": f"DeepSeek, if your {animal} is experiencing {symptoms_text}, what is an insightful and varied question a pet owner might ask to a veterinarian? give me a short and diverse question."}
            ],
            "max_tokens": 50,
            "temperature": 1.2,  # เพิ่มความหลากหลายให้กับคำถาม
            "top_p": 0.9  # จำกัดความเป็นไปได้ของคำถามที่ออกมา
        }

        # เรียกใช้ API
        response = requests.post(API_URL, headers=headers, json=data)

        # ตรวจสอบว่าการเรียก API สำเร็จหรือไม่
        if response.status_code == 200:
            result = response.json()
            question = result["choices"][0]["message"]["content"]

            # ตรวจสอบคำถามซ้ำ
            first_word = question.split()[0].lower()  # หาคำแรกของคำถามและแปลงเป็นตัวพิมพ์เล็ก
            if first_word not in previous_first_words:
                # บันทึกคำถามใหม่ในไฟล์ CSV
                writer.writerow([question])
                previous_first_words.append(first_word)  # เก็บคำแรกของคำถามในลิสต์

                # แสดงผลลัพธ์ใน terminal (optional)
                print(f"Iteration {i + 1}:")
                print(f"Animal: {animal}")
                print(f"Symptoms: {symptoms_text}")
                print(f"Question: {question}")
                print("-" * 50)
            else:
                print(f"Iteration {i + 1}: Skipped question starting with forbidden word '{first_word}'.")

        # หากลิสต์มีคำแรกเกิน 10 คำแล้ว ให้ลบค่าในลิสต์
        if len(previous_first_words) > 7:
            previous_first_words.clear()  # ลบคำทั้งหมดในลิสต์

        # พิมพ์ค่าของ previous_first_words ในแต่ละลูป
        print(f"previous_first_words after iteration {i + 1}: {previous_first_words}")

print(f"บันทึกคำถามลงในไฟล์ {csv_file} เสร็จสิ้น")
