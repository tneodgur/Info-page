import streamlit as st
import time
import csv
import io
from google import genai
from google.genai.errors import APIError

# 0) 기본 설정
st.set_page_config(page_title="Gemini 고객 응대 챗봇", layout="wide")

# 1) 모델/시스템 지침
AVAILABLE_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-pro",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]

SYSTEM_INSTRUCTION = (
    "당신은 고객의 장비 대여 가능 여부 문의 및 불편 사항을 접수하는 전문 고객 응대 챗봇입니다. "
    "사용자의 상황에 깊이 공감하고, 매우 정중하며 친절한 말투로 응대해야 합니다.\n\n"
    "응대 규칙:\n"
    "1. 사용자의 불편 사항이나 장비 대여 문의에 대해 공감하고 정중하게 응답하십시오.\n"
    "2. 답변 시, 사용자의 불편 사항 또는 장비 대여 요청 내용을 '무엇이', '언제', '어디서', '어떻게'에 맞춰 요약하고, "
    "담당자에게 전달해 신속히 처리하겠다고 안내하십시오.\n"
    "3. 모든 응답의 마지막에는 담당자 확인 후 회신을 위해 반드시 사용자의 이메일 주소를 정중하게 요청하십시오.\n"
    "4. 사용자가 이메일 제공을 명시적으로 거부하면 다음 문구를 그대로 사용하십시오: "
    "\"죄송하지만, 연락처 정보를 받지 못하여 담당자의 검토 내용을 받으실 수 없어요.\""
)

# 2) 세션 상태
def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []  # {'role': 'user'|'assistant', 'parts': [{'text': str}]}
    if "model_name" not in st.session_state:
        st.session_state.model_name = AVAILABLE_MODELS[0]
    if "log_history" not in st.session_state:
        st.session_state.log_history = []
    if "enable_logging" not in st.session_state:
        st.session_state.enable_logging = False
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"sess-{time.time()}"

init_state()

# 3) 사이드바
with st.sidebar:
    st.header("설정")

    # API 키
    st.session_state.api_key = None
    try:
        if "GEMINI_API_KEY" in st.secrets:
            st.session_state.api_key = st.secrets["GEMINI_API_KEY"]
            st.success("API 키 로드 완료 (st.secrets)")
    except Exception:
        pass

    if st.session_state.api_key is None:
        st.session_state.api_key = st.text_input(
            "Gemini API Key", type="password", placeholder="키가 없으면 동작하지 않습니다."
        )

    # 모델 선택
    st.session_state.model_name = st.selectbox(
        "모델 선택", AVAILABLE_MODELS, index=AVAILABLE_MODELS.index(st.session_state.model_name)
        if st.session_state.model_name in AVAILABLE_MODELS else 0
    )

    # 로깅
    st.session_state.enable_logging = st.checkbox(
        "대화 CSV 기록", value=st.session_state.enable_logging
    )

    st.markdown("---")

    # 초기화
    if st.button("대화 초기화", type="primary"):
        st.session_state.messages = []
        st.session_state.log_history = []
        st.rerun()

    # 로그 다운로드
    st.subheader("세션")
    st.write(f"모델: **{st.session_state.model_name}**")
    st.write(f"세션 ID: **{st.session_state.session_id}**")

    if st.session_state.log_history:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=["timestamp", "role", "content"])
        writer.writeheader()
        for r in st.session_state.log_history:
            writer.writerow(r)
        st.download_button("대화 로그 다운로드", buf.getvalue(), "chat_log.csv", "text/csv")
    else:
        st.info("저장된 대화 없음")

# 4) 유틸
def log_message(role: str, content: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    if st.session_state.enable_logging:
        st.session_state.log_history.append({"timestamp": ts, "role": role, "content": content})
    st.session_state.messages.append({"role": role, "parts": [{"text": content}]})

def build_api_history():
    """
    SDK가 system/system_instruction을 지원하지 않는 경우를 대비해,
    시스템 지침을 '첫 번째 user 메시지'로 앞에 붙여 보냅니다.
    또한 assistant→model로 역할을 매핑합니다.
    """
    conv = []
    for m in st.session_state.messages:
        role = "model" if m["role"] == "assistant" else "user"
        conv.append({"role": role, "parts": m["parts"]})
    # 맨 앞에 시스템 지침을 user로 삽입
    return [{"role": "user", "parts": [{"text": SYSTEM_INSTRUCTION}]}] + conv

def call_api(client, model, contents):
    try:
        return client.models.generate_content(model=model, contents=contents)
    except APIError as e:
        st.error(f"API 오류: {e}")
    except Exception as e:
        st.error(f"예상치 못한 오류: {e}")
    return None

# 5) 메인
st.title("🌟 AI 고객 응대 센터 챗봇")
st.caption("장비 대여 문의 및 불편 사항을 접수해 드립니다.")

# 과거 메시지 표시
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["parts"][0]["text"])

# 입력 처리
user_prompt = st.chat_input("문의 사항을 입력해 주세요.")
if user_prompt:
    with st.chat_message("user"):
        st.markdown(user_prompt)
    log_message("user", user_prompt)

    if not st.session_state.api_key:
        st.error("Gemini API Key를 입력해 주세요.")
    else:
        client = genai.Client(api_key=st.session_state.api_key)
        history = build_api_history()
        if len(history) > 6:  # 최근 3턴만 유지
            history = history[-6:]

        with st.spinner(f"({st.session_state.model_name}) 모델이 답변을 생성하는 중..."):
            resp = call_api(client, st.session_state.model_name, history)

        if resp:
            answer = resp.text
            with st.chat_message("assistant"):
                st.markdown(answer)
            log_message("assistant", answer)
        else:
            st.error("응답 생성 실패")


