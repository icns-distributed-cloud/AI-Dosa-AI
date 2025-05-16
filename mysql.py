from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

load_dotenv() 

# MySQL 데이터베이스 설정
DATABASE_URL = os.getenv("DATABASE_URL")

# 데이터베이스 연결 엔진 생성
engine = create_engine(DATABASE_URL)

# 세션 설정
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 데이터베이스 세션 가져오는 함수
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ✅ MySQL 연결 테스트
if __name__ == "__main__":
    try:
        db = next(get_db())  # 데이터베이스 세션 열기
        print("✅ MySQL 연결 성공!")
    except Exception as e:
        print("❌ MySQL 연결 실패:", e)
