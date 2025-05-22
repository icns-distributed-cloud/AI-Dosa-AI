from fastapi import FastAPI, UploadFile, File, Form, Depends
import shutil
import os
import requests
import time
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from sqlalchemy import Column, Integer, String, TIMESTAMP, ForeignKey, Text, func
from sqlalchemy.ext.declarative import declarative_base
from mysql import get_db
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
import openai

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인에서 요청 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용 (GET, POST, PUT, DELETE 등)
    allow_headers=["*"],  # 모든 헤더 허용
)

load_dotenv() 

# STT 모델 로드
openai.api_key = os.getenv("OPENAI_API_KEY")

# 디렉토리 설정
INPUT_DIR = os.getenv("INPUT_DIR")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

Base = declarative_base()

MBTI_VECTOR_DB_PATH = os.getenv("MBTI_VECTOR_DB_PATH")
MBTI_DATA_PATH = "./mbti_data.txt"

example_dirs = ["./example1", "./example2", "./example3"]
file_counter = 0  # 순서 기억용 전역변수

def save_file_and_dirs(file: UploadFile, meeting_id: int):
    input_meeting_dir = os.path.join(INPUT_DIR, str(meeting_id))
    output_meeting_dir = os.path.join(OUTPUT_DIR, str(meeting_id))
    os.makedirs(input_meeting_dir, exist_ok=True)
    os.makedirs(output_meeting_dir, exist_ok=True)

    file_path = os.path.join(input_meeting_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return file_path, output_meeting_dir

def save_transcription(output_dir: str, filename: str, text: str):
    output_filename = os.path.splitext(filename)[0] + ".txt"
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

# Bot 테이블 모델 정의
class Bot(Base):
    __tablename__ = "bots"
    bot_id = Column(Integer, primary_key=True, autoincrement=True)
    meeting_id = Column(Integer, nullable=False)
    type = Column(String, nullable=True)  # type 컬럼 추가
    content = Column(String, nullable=True)
    created_at = Column(TIMESTAMP(6), server_default=func.now())
    
# user_meetings 테이블 엔티티
class UserMeeting(Base):
    __tablename__ = "user_meetings"

    user_meeting_id = Column(Integer, primary_key=True, autoincrement=True)
    entry_time = Column(TIMESTAMP(6), nullable=False)
    exit_time = Column(TIMESTAMP(6), nullable=True)
    meeting_id = Column(Integer, nullable=False)
    user_id = Column(Integer, nullable=False)
    user_team_id = Column(Integer, nullable=False)
    
# note 테이블 엔티티
class Note(Base):
    __tablename__ = "note"

    note_id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(TIMESTAMP(6), server_default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP(6), onupdate=func.now(), nullable=True)
    members = Column(String(255), nullable=True)
    audio_url = Column(String(1000), nullable=True)
    script = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    title = Column(String(255), nullable=True)
    meeting_id = Column(Integer, ForeignKey("user_meetings.meeting_id"), nullable=False)
    team_id = Column(Integer, ForeignKey("meetings.team_id"), nullable=False) 

# users 테이블 엔티티
class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(TIMESTAMP(6), server_default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP(6), onupdate=func.now(), nullable=True)
    email = Column(String(255), unique=True, nullable=False)
    nickname = Column(String(255), nullable=False)
    password = Column(String(255), nullable=False)
    profile = Column(String(2048), nullable=True)

# meetings 테이블 엔티티 추가
class Meeting(Base):
    __tablename__ = "meetings"

    meeting_id = Column(Integer, primary_key=True, autoincrement=True)
    duration = Column(Integer, nullable=True)  # NULL 가능
    ended_at = Column(TIMESTAMP(6), nullable=True)
    started_at = Column(TIMESTAMP(6), nullable=False)
    title = Column(String(255), nullable=True)
    team_id = Column(Integer, nullable=False)

    

LLM_IP = os.getenv("LLM_IP")

# LLM 서버 API URL
LLM_API_URLS = {
    "POSITIVE": f"{LLM_IP}/api/bot/positive",
    "NEGATIVE": f"{LLM_IP}/api/bot/negative",
    "SUMMARY": f"{LLM_IP}/api/bot/summary",
    "LOADER": f"{LLM_IP}/api/bot/loader",
    "END": f"{LLM_IP}/api/bot/endmeeting",
    "FOOD": f"{LLM_IP}/api/bot/food",
    "SAJU": f"{LLM_IP}/api/bot/saju",
    "GYM": f"{LLM_IP}/api/bot/gym",
    "MOYA": f"{LLM_IP}/api/bot/moya",
}

# LLM 서버에 STT 결과 전달하는 함수
def send_to_llm(llm_url, text, meeting_id):
    payload = {"script": text,
               "meeting_id": meeting_id}
    headers = {"Content-Type": "application/json"}
    response = requests.post(llm_url, json=payload, headers=headers)
    return response.json().get("response", "응답을 가져올 수 없습니다.")

# RAG(LOADER) 전용 LLM 서버 요청 함수 (stt_text + scripts 리스트 전송)
def send_to_llm_loader(llm_url, stt_text, scripts, meeting_id):
    payload = {"stt_text": stt_text, "scripts": scripts, "meeting_id": meeting_id}
    headers = {"Content-Type": "application/json"}
    response = requests.post(llm_url, json=payload, headers=headers)
    return response.json().get("response", "응답을 가져올 수 없습니다.")

def whisper_api_transcribe(file_path):
    with open(file_path, "rb") as audio_file:
        response = openai.Audio.transcribe(
            model="whisper-1",
            file=audio_file,
            response_format="json"
        )
    return response["text"]

@app.post("/api/positive")
async def transcribe_positive(
    file: UploadFile = File(...),
    meeting_id: int = Form(...),
    db: Session = Depends(get_db)
):
    
    file_path, output_dir = save_file_and_dirs(file, meeting_id)

    # Insert placeholder in DB
    new_bot_entry = Bot(
        meeting_id=meeting_id,
        type="POSITIVE",
        content="요약중",
        created_at=func.now()
    )
    db.add(new_bot_entry)
    db.commit()
    db.refresh(new_bot_entry)

    # STT
    start_time = time.time()
    text = whisper_api_transcribe(file_path)
    end_time = time.time()
    print(f"STT 처리 시간: {end_time - start_time:.2f}초")

    save_transcription(output_dir, file.filename, text)

    # LLM 처리
    llm_response = send_to_llm(LLM_API_URLS["POSITIVE"], text, meeting_id)
    new_bot_entry.content = llm_response
    db.commit()

    return {"transcription": text, "llm_response": llm_response}

@app.post("/api/moya")
async def transcribe_summary(
    file: UploadFile = File(...),
    meeting_id: int = Form(...),
    db: Session = Depends(get_db)
):

    file_path, output_dir = save_file_and_dirs(file, meeting_id)
    
    new_bot_entry = Bot(
        meeting_id=meeting_id,
        type="MOYA",
        content="대화중",
        created_at=func.now()
    )
    db.add(new_bot_entry)
    db.commit()
    db.refresh(new_bot_entry)
    
    start_time = time.time() #stt 시작 시간
    text = whisper_api_transcribe(file_path)
    end_time = time.time() #stt 종료 시간
    processing_time = end_time - start_time #걸린 시간
    print(f"Request time: {processing_time} seconds")
        
    save_transcription(output_dir, file.filename, text)

    # 모든 .txt 병합
    merged_text = ""
    print(f"[MOYA] 텍스트 병합 시작 - 디렉토리: {output_dir}")
    for filename in os.listdir(output_dir):
        if filename.endswith(".txt"):
            full_path = os.path.join(output_dir, filename)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    merged_text += content.strip() + "\n\n"
                    print(f"[MOYA] 병합된 파일: {filename} (길이: {len(content)}자)")
            except Exception as e:
                print(f"[WARNING] 병합 실패: {filename} - {e}")

    print(f"[MOYA] 최종 병합된 텍스트 길이: {len(merged_text)}자")
    print(f"[MOYA] 병합된 전체 텍스트: {merged_text}")

    llm_response = send_to_llm(LLM_API_URLS["MOYA"], merged_text.strip(), meeting_id)
    
    new_bot_entry.content = llm_response
    db.commit()

    return {"transcription": text, "llm_response": llm_response}

@app.post("/api/negative")
async def transcribe_negative(
    file: UploadFile = File(...),
    meeting_id: int = Form(...),
    db: Session = Depends(get_db)
):

    file_path, output_dir = save_file_and_dirs(file, meeting_id)
    
    new_bot_entry = Bot(
        meeting_id=meeting_id,
        type="NEGATIVE",
        content="요약중",
        created_at=func.now()
    )
    db.add(new_bot_entry)
    db.commit()
    db.refresh(new_bot_entry)
    
    start_time = time.time() #stt 시작 시간
    text = whisper_api_transcribe(file_path)
    end_time = time.time() #stt 종료 시간
    processing_time = end_time - start_time #걸린 시간
    print(f"Request time: {processing_time} seconds")

    save_transcription(output_dir, file.filename, text)

    llm_response = send_to_llm(LLM_API_URLS["NEGATIVE"], text, meeting_id)
    
    new_bot_entry.content = llm_response
    db.commit()
    
    return {"transcription": text, "llm_response": llm_response}

@app.post("/api/summary")
async def transcribe_summary(
    file: UploadFile = File(...),
    meeting_id: int = Form(...),
    db: Session = Depends(get_db)
):

    file_path, output_dir = save_file_and_dirs(file, meeting_id)
    
    new_bot_entry = Bot(
        meeting_id=meeting_id,
        type="SUMMARY",
        content="요약중",
        created_at=func.now()
    )
    db.add(new_bot_entry)
    db.commit()
    db.refresh(new_bot_entry)
    
    start_time = time.time() #stt 시작 시간
    text = whisper_api_transcribe(file_path)
    end_time = time.time() #stt 종료 시간
    processing_time = end_time - start_time #걸린 시간
    print(f"Request time: {processing_time} seconds")
        
    save_transcription(output_dir, file.filename, text)

    # 모든 .txt 병합
    merged_text = ""
    print(f"[SUMMARY] 텍스트 병합 시작 - 디렉토리: {output_dir}")
    for filename in os.listdir(output_dir):
        if filename.endswith(".txt"):
            full_path = os.path.join(output_dir, filename)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    merged_text += content.strip() + "\n\n"
                    print(f"[SUMMARY] 병합된 파일: {filename} (길이: {len(content)}자)")
            except Exception as e:
                print(f"[WARNING] 병합 실패: {filename} - {e}")

    print(f"[SUMMARY] 최종 병합된 텍스트 길이: {len(merged_text)}자")
    print(f"[SUMMARY] 병합된 전체 텍스트: {merged_text}")

    llm_response = send_to_llm(LLM_API_URLS["SUMMARY"], merged_text.strip(), meeting_id)
    
    new_bot_entry.content = llm_response
    db.commit()

    return {"transcription": text, "llm_response": llm_response}

@app.post("/api/loader")
async def transcribe_loader(
    file: UploadFile = File(...),
    meeting_id: int = Form(...),
    db: Session = Depends(get_db)
):
    # 파일 저장
    file_path, output_dir = save_file_and_dirs(file, meeting_id)

    # bot 테이블에 기록
    new_bot_entry = Bot(meeting_id=meeting_id, type="LOADER", content="분석중...", created_at=func.now())
    db.add(new_bot_entry)
    db.commit()
    db.refresh(new_bot_entry)

    # STT 변환 수행
    start_time = time.time()
    stt_text = whisper_api_transcribe(file_path)
    print(f"STT 처리 시간: {time.time() - start_time:.2f}초")

    save_transcription(output_dir, file.filename, stt_text)

    # meetingId로 teamId 조회
    meeting = db.query(Meeting).filter(Meeting.meeting_id == meeting_id).first()
    if not meeting:
        return {"error": "Meeting not found"}

    team_id = meeting.team_id

    # teamId에 매칭되는 noteId 및 script 조회
    notes = db.query(Note).filter(Note.team_id == team_id).all()
    note_ids = [note.note_id for note in notes]
    scripts = [note.script for note in notes]

    # RAG 수행을 위한 새로운 send_to_llm_loader 사용
    print(f"[STT] Sending to LLM: stt_text={stt_text[:100]}, scripts={scripts[:2]}")  # 로그 추가
    llm_response = send_to_llm_loader(LLM_API_URLS["LOADER"], stt_text, scripts, meeting_id)

    # bot 테이블에 LLM 응답 저장
    new_bot_entry.content = llm_response
    db.commit()

    return {"note_ids": note_ids[0], "response": llm_response}

@app.post("/api/food")
async def transcribe_summary(
    file: UploadFile = File(...),
    meeting_id: int = Form(...),
    db: Session = Depends(get_db)
):

    file_path, output_dir = save_file_and_dirs(file, meeting_id)
    
    new_bot_entry = Bot(
        meeting_id=meeting_id,
        type="FOOD",
        content="음식 분석중",
        created_at=func.now()
    )
    db.add(new_bot_entry)
    db.commit()
    db.refresh(new_bot_entry)
    
    start_time = time.time() #stt 시작 시간
    text = whisper_api_transcribe(file_path)
    end_time = time.time() #stt 종료 시간
    processing_time = end_time - start_time #걸린 시간
    print(f"Request time: {processing_time} seconds")
        
    save_transcription(output_dir, file.filename, text)

    # 모든 .txt 병합
    merged_text = ""
    print(f"[FOOD] 텍스트 병합 시작 - 디렉토리: {output_dir}")
    for filename in os.listdir(output_dir):
        if filename.endswith(".txt"):
            full_path = os.path.join(output_dir, filename)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    merged_text += content.strip() + "\n\n"
                    print(f"[FOOD] 병합된 파일: {filename} (길이: {len(content)}자)")
            except Exception as e:
                print(f"[WARNING] 병합 실패: {filename} - {e}")

    print(f"[FOOD] 최종 병합된 텍스트 길이: {len(merged_text)}자")
    print(f"[FOOD] 병합된 전체 텍스트: {merged_text}")

    llm_response = send_to_llm(LLM_API_URLS["FOOD"], merged_text.strip(), meeting_id)
    
    new_bot_entry.content = llm_response
    db.commit()

    return {"transcription": text, "llm_response": llm_response}

@app.post("/api/saju")
async def transcribe_saju(
    file: UploadFile = File(...),
    meeting_id: int = Form(...),
    db: Session = Depends(get_db)
):

    file_path, output_dir = save_file_and_dirs(file, meeting_id)

    new_bot_entry = Bot(
        meeting_id=meeting_id,
        type="SAJU",
        content="사주 분석중...",
        created_at=func.now()
    )
    db.add(new_bot_entry)
    db.commit()
    db.refresh(new_bot_entry)

    start_time = time.time() #stt 시작 시간
    stt_text = whisper_api_transcribe(file_path)
    end_time = time.time()
    print(f"STT 처리 시간: {end_time - start_time:.2f}초")

    save_transcription(output_dir, file.filename, stt_text)

    llm_response = send_to_llm(LLM_API_URLS["SAJU"], stt_text, meeting_id)

    new_bot_entry.content = llm_response
    db.commit()

    return {"transcription": stt_text, "llm_response": llm_response}

@app.post("/api/gym")
async def transcribe_summary(
    file: UploadFile = File(...),
    meeting_id: int = Form(...),
    db: Session = Depends(get_db)
):

    file_path, output_dir = save_file_and_dirs(file, meeting_id)
    
    new_bot_entry = Bot(
        meeting_id=meeting_id,
        type="GYM",
        content="운동 분석중",
        created_at=func.now()
    )
    db.add(new_bot_entry)
    db.commit()
    db.refresh(new_bot_entry)
    
    start_time = time.time() #stt 시작 시간
    text = whisper_api_transcribe(file_path)
    end_time = time.time() #stt 종료 시간
    processing_time = end_time - start_time #걸린 시간
    print(f"Request time: {processing_time} seconds")
        
    save_transcription(output_dir, file.filename, text)

    # 모든 .txt 병합
    merged_text = ""
    print(f"[GYM] 텍스트 병합 시작 - 디렉토리: {output_dir}")
    for filename in os.listdir(output_dir):
        if filename.endswith(".txt"):
            full_path = os.path.join(output_dir, filename)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    merged_text += content.strip() + "\n\n"
                    print(f"[GYM] 병합된 파일: {filename} (길이: {len(content)}자)")
            except Exception as e:
                print(f"[WARNING] 병합 실패: {filename} - {e}")

    print(f"[GYM] 최종 병합된 텍스트 길이: {len(merged_text)}자")
    print(f"[GYM] 병합된 전체 텍스트: {merged_text}")

    llm_response = send_to_llm(LLM_API_URLS["GYM"], merged_text.strip(), meeting_id)
    
    new_bot_entry.content = llm_response
    db.commit()

    return {"transcription": text, "llm_response": llm_response}

@app.post("/api/v1/endmeeting")
async def end_meeting(
    file: UploadFile = File(...),
    meeting_id: int = Form(...),
    db: Session = Depends(get_db)
):

    # 파일 저장 (MP3)
    file_path, output_dir = save_file_and_dirs(file, meeting_id)

    # userMeetings 테이블에서 meeting_id에 해당하는 user_id 조회
    user_ids = db.query(UserMeeting.user_id).filter(UserMeeting.meeting_id == meeting_id).all()
    user_ids = [user_id[0] for user_id in user_ids]

    # users 테이블에서 nickname 조회하여 members 필드에 저장
    nicknames = db.query(User.nickname).filter(User.user_id.in_(user_ids)).all()
    members = ", ".join([nickname[0] for nickname in nicknames])

    # meetings 테이블에서 team_id 조회
    meeting = db.query(Meeting).filter(Meeting.meeting_id == meeting_id).first()
    if not meeting:
        return {"error": "Meeting not found"}

    team_id = meeting.team_id  # team_id 값 저장

    # STT 변환 수행 (script 저장)
    text = whisper_api_transcribe(file_path)
    save_transcription(output_dir, file.filename, text)

    # Ollama로 요약 요청 (summary 저장)
    llm_response = send_to_llm(LLM_API_URLS["END"], text, meeting_id)

    print(f"LLM 응답: {llm_response}...")  # 처음 100자만 출력하여 확인

    # 현재 날짜 (YYYY-MM-DD 형식)
    current_date = datetime.now().strftime("%Y-%m-%d")

    # note 테이블에 데이터 추가 (team_id 포함)
    new_note = Note(
        created_at=func.now(),
        updated_at=func.now(),
        members=members,
        audio_url=file_path,
        script=text,
        summary=llm_response,
        title=current_date,
        meeting_id=meeting_id,
        team_id=team_id  # team_id 추가
    )
    db.add(new_note)

    # meetings 테이블 업데이트 (ended_at & duration 추가)
    ended_at = datetime.now()
    meeting.ended_at = ended_at

    if meeting.started_at:
        if ended_at < meeting.started_at:
            meeting.duration = 0  # 잘못된 경우 0 설정
        else:
            # 나노초 단위로 저장
            duration_ns = int((ended_at - meeting.started_at).total_seconds() * 1e9)
            meeting.duration = duration_ns
    else:
        meeting.duration = 0  # started_at이 없으면 0

    db.commit()

    return

@app.post("/api/v1/endtest")
async def end_meeting(
    file: UploadFile = File(...),
    meeting_id: int = Form(...),
    db: Session = Depends(get_db)
):

    global file_counter

    # 이미 해당 meeting_id로 note가 존재하면 중복 생성 방지
    existing_note = db.query(Note).filter(Note.meeting_id == meeting_id).first()
    if existing_note:
        print(f"[ENDTEST] 이미 존재하는 meeting_id: {meeting_id} — note 추가 생략")
        return {"message": "이미 이 meeting_id에 대한 note가 존재합니다."}
    
    # 1. 예시 디렉토리 순환
    selected_dir = example_dirs[file_counter]
    file_counter = (file_counter + 1) % len(example_dirs)

    script_path = os.path.join(selected_dir, "script.txt")
    summary_path = os.path.join(selected_dir, "summary.txt")

    print(f"[ENDTEST] 선택된 예시 디렉토리: {selected_dir}")
    print(f"[ENDTEST] 스크립트 경로: {script_path}")
    print(f"[ENDTEST] 요약문 경로: {summary_path}")

    try:
        with open(script_path, "r", encoding="utf-8") as f:
            script_text = f.read()
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_text = f.read()
    except Exception as e:
        return {"error": f"예시 파일 읽기 실패: {e}"}

    # 파일 저장 (MP3)
    file_path, output_dir = save_file_and_dirs(file, meeting_id)

    # userMeetings 테이블에서 meeting_id에 해당하는 user_id 조회
    user_ids = db.query(UserMeeting.user_id).filter(UserMeeting.meeting_id == meeting_id).all()
    user_ids = [user_id[0] for user_id in user_ids]

    # users 테이블에서 nickname 조회하여 members 필드에 저장
    nicknames = db.query(User.nickname).filter(User.user_id.in_(user_ids)).all()
    members = ", ".join([nickname[0] for nickname in nicknames])

    # meetings 테이블에서 team_id 조회
    meeting = db.query(Meeting).filter(Meeting.meeting_id == meeting_id).first()
    if not meeting:
        return {"error": "Meeting not found"}

    team_id = meeting.team_id  # team_id 값 저장


    # 현재 날짜 (YYYY-MM-DD 형식)
    current_date = datetime.now().strftime("%Y-%m-%d")

    # note 테이블에 데이터 추가 (team_id 포함)
    new_note = Note(
        created_at=func.now(),
        updated_at=func.now(),
        members=members,
        audio_url=file_path,
        script=script_text,
        summary=summary_text,
        title=current_date,
        meeting_id=meeting_id,
        team_id=team_id  # team_id 추가
    )
    db.add(new_note)

    # meetings 테이블 업데이트 (ended_at & duration 추가)
    ended_at = datetime.now()
    meeting.ended_at = ended_at

    if meeting.started_at:
        if ended_at < meeting.started_at:
            meeting.duration = 0  # 잘못된 경우 0 설정
        else:
            # 나노초 단위로 저장
            duration_ns = int((ended_at - meeting.started_at).total_seconds() * 1e9)
            meeting.duration = duration_ns
    else:
        meeting.duration = 0  # started_at이 없으면 0

    db.commit()

    return