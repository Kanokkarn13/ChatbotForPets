import asyncio
import aiohttp
import csv
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
file_path = "qq2.csv"
df = pd.read_csv(file_path)

# จำกัดจำนวนคำขอพร้อมกัน
SEMAPHORE = asyncio.Semaphore(10)

# ตั้งค่าไฟล์ CSV สำหรับบันทึกคำตอบ
csv_file = "pet_answers_v2.csv"


async def fetch_answer(session, question):
    """ส่งคำขอ API แบบ asynchronous"""
    async with SEMAPHORE:
        data = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "I am a chatbot specialized in advising pet owners about their animals' symptoms. Please answer the following questions in short, conversational sentences. Do not recommend visiting a vet immediately. Keep responses under 70 tokens."},
                {"role": "user", "content": question}
            ],
            "max_tokens": 100,
            "temperature": 1.0,
            "top_p": 0.9
        }

        try:
            async with session.post(API_URL, headers=HEADERS, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    answer = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                    return question, answer
                else:
                    print(f"Error {response.status}: {await response.text()}")
                    return question, "Error fetching answer"
        except aiohttp.ClientError as e:
            print(f"Request failed: {e}")
            return question, "Request failed"


async def main():
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(8000,9110):  # วนลูป 1000 รอบ
            if i >= len(df):  # หยุดถ้าเกินจำนวนคำถามที่มี
                break
            row = df.iloc[i]  # อ่านคำถามตามลำดับ
            question = row["Question"]
            tasks.append(fetch_answer(session, question))

        results = await asyncio.gather(*tasks)

        # บันทึกผลลัพธ์ลง CSV
        with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Question", "Answer"])
            writer.writerows(results)

    print(f"บันทึกคำตอบลงในไฟล์ {csv_file} เสร็จสิ้น")



asyncio.run(main())
