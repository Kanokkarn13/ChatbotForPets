import asyncio
import aiohttp
import csv
import random
import pandas as pd
import os
from dotenv import load_dotenv

# โหลดตัวแปรแวดล้อม
load_dotenv()
API_KEY = os.getenv("DeepseekAPIkey")
API_URL = "https://api.deepseek.com/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# โหลด dataset
file_path = "dataset/data.csv"
df = pd.read_csv(file_path)

# จำกัดจำนวนคำขอพร้อมกัน
SEMAPHORE = asyncio.Semaphore(10)

# เก็บคำถามที่เคยสร้างไปแล้ว
previous_questions = set()

# Prompt Templates เพื่อเพิ่มความหลากหลาย
QUESTION_TEMPLATES = [
    "General symptoms",
    "Treatment",
    "Prevention",
    "Specific behaviors and causes",
    "More information",
    "Emergency signs",
    "Common illnesses",
    "Vaccination schedule",
    "Exercise and activity levels",
]

async def fetch_question(session, animal, symptoms):
    """ส่งคำขอ API แบบ asynchronous และสร้างคำถามแบบหลากหลาย"""
    async with SEMAPHORE:
        symptoms_text = " and ".join(symptoms)
        question_prompt = random.choice(QUESTION_TEMPLATES)

        data = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": f"You are a pet owner with no knowledge about animal diseases. You are worried about your pet and want to ask a veterinarian for advice in a natural, non-technical way.Keep responses under 50 tokens."
                },
                {
                    "role": "user",
                    "content": f"DeepSeek, your pet is {animal} and It is experiencing {symptoms}, what is an insightful and varied question a pet owner might ask for {question_prompt} Provide a short, unique question. Only respond with a question."
                }
            ],
            "max_tokens": 70,
            "temperature": 2.0,
            "top_p": 0.9
        }

        try:
            async with session.post(API_URL, headers=HEADERS, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    question = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                    
                    # ตรวจสอบไม่ให้ซ้ำ
                    if question in previous_questions:
                        print(f"Skipping duplicate: {question}")
                        return None
                    
                    previous_questions.add(question)
                    return question
                else:
                    print(f"Error {response.status}: {await response.text()}")
                    return None
        except aiohttp.ClientError as e:
            print(f"Request failed: {e}")
            return None

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = []
        for _ in range(1000):
            row = df.sample(n=1).iloc[0]
            animal = row["AnimalName"]
            symptoms = row[1:-1].dropna().tolist()
            num_symptoms = random.randint(1, min(3, len(symptoms)))
            selected_symptoms = random.sample(symptoms, num_symptoms) if symptoms else ["unknown symptom"]
            
            tasks.append(fetch_question(session, animal, selected_symptoms))  # ✅ รวม task async
        
        results = await asyncio.gather(*tasks)  # ✅ ทำงานแบบ async พร้อมกัน
        
        # บันทึกผลลัพธ์ลงไฟล์
        with open("pet_questions_v3.csv", mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            for question in results:
                if question:
                    writer.writerow([question])
    
    print(f"บันทึกคำถามลงในไฟล์ pet_questions_v3.csv เสร็จสิ้น")

asyncio.run(main())
