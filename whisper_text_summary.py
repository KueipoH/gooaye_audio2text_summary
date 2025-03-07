import whisper
import os
import glob
from openai import OpenAI
import torch
import time
from datetime import datetime

def format_time(seconds):
    """格式化時間顯示"""
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def extract_info():
    try:
        # 確認 GPU 狀態
        if not torch.cuda.is_available():
            print("警告: 未檢測到 GPU，這可能會大幅降低處理速度")
            return
            
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
        print(f"當前 CUDA 版本: {torch.version.cuda}")
        
        # 載入 medium 模型
        print("\n正在載入 Whisper turbo 模型...")
        model = whisper.load_model("turbo", device="cuda")
        print("模型載入完成")
        
        # 找到 MP4 檔案
        videos = glob.glob("/home/a/Downloads/EP537.mp4")
        if not videos:
            print("找不到 MP4 檔案")
            return
            
        video_path = videos[0]
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        current_date = datetime.now().strftime("%Y-%m-%d")
        output_filename = f"{video_name}-{current_date}.txt"
        
        # 檢查是否已存在轉錄檔案
        if os.path.exists(output_filename):
            response = input(f"檔案 {output_filename} 已存在。是否要重新轉錄？(y/n): ")
            if response.lower() != 'y':
                print("程式結束")
                return
        
        print(f"\n開始處理影片: {os.path.basename(video_path)}")
        
        # 記錄開始時間
        start_time = time.time()
        
        # 使用 Whisper 進行轉錄，增加中文相關參數
        print("\n正在轉錄影片...")
        result = model.transcribe(
            video_path,
            language="zh",
            task="transcribe",
            initial_prompt="這是一段關於股票和產業分析的影片。",
            best_of=2,  # 使用多次解碼提高準確度
            fp16=True,  # 使用半精度加速
            verbose=True  # 顯示詳細進度
        )
        
        transcript = result["text"]
        print(f"\n轉錄完成，用時: {format_time(time.time() - start_time)}")
        
        # 使用 GPT 提取重要資訊
        print("\n正在分析內容...")
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            print("錯誤: 找不到 OPENAI_API_KEY 環境變數")
            return
            
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """請從以下文字中提取所有提到的股票和產業資訊，格式如下：

                    請從以下文字中提取所有提到的股票和產業資訊，格式如下：

                    股票資訊：
                    - [股票代號] 公司名稱：提及內容摘要

                    產業資訊：
                    - 產業名稱：相關描述或分析重點

                    注意：
                    1. 只列出確實提到的資訊
                    2. 盡可能保留原文用詞
                    3. 如果同一個股票/產業多次提及，合併為一條並保留重要內容

                    如果有額外重要市場資訊也請提供 lets do it step by step"""
                },
                {
                    "role": "user",
                    "content": transcript
                }
            ],
            temperature=0.3
        )
        
        print("\n=== 分析結果 ===")
        analysis_result = response.choices[0].message.content
        print(analysis_result)
        
        # 儲存分析結果，使用新的檔名格式
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write("=== 原始轉錄文字 ===\n\n")
            f.write(transcript)
            f.write("\n\n=== 分析結果 ===\n\n")
            f.write(analysis_result)
            
        print(f"\n分析結果已儲存至 {output_filename}")
        
    except Exception as e:
        print(f"\n處理時發生錯誤: {str(e)}")
        
    finally:
        # 清理 GPU 記憶體
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    extract_info()