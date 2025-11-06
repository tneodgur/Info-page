import streamlit as st
import time
import csv
import io
from google import genai
from google.genai.errors import APIError

# ─────────────────────────────────────────────────────────────
# 0) Streamlit 기본 설정
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Gemini 고객 응대 챗봇", layout="wide")

# ─────────────────────────────────────────────────────────────
# 1) 환경 설정 및 시스템 프롬프트
# ─────────────────────────────────────────────────────────────
AVAILABLE_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.0-pro",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]

SYSTEM_INSTRUCTION = """
당신은 고객의 장비 대여 가능 여부 문의 및 불편 사항을 접수하는 전문 고객 응대 챗봇입니다. 
사용자의 상황에 깊이 공감하고, 매우 정중하며 친절한 말투로 응대해야 합니다.

응대 규칙:
1. 사용자의 불편 사항이나 장비 대여 문의에 대해 공감하고 정중하게 응답하십시오.
2. 답변 시, 사용자의 불편 사항 또는 장비 대여 요청 내용을 '무엇이', '언제', '어디서', '어떻게'에 맞춰 구체적으로 요약하고, 이를 고객 응대 담당자에게 전달하여 신속히 처리하겠다는 취지로 안내해야 합니다.
3. 모든 응답의 마지막에는 담당자 확인 후 회신을 위해 반드시 사용자의 이메일 주소를 정중하게 요청하십시오.
4. 만일 사용자가 이메일 주소 제공을 명시적으로 거부할 경우, 다음 문구를 그대로 사용하여 정중히 안내하십시오: "죄송하지만, 연락처 정보를 받지 못하여 담당자의 검토 내용을 받으실 수 없어요."
"""

# ─────────────────────────────────────────────────────────────
# 2) 세션 상태 초기화
# ─────────────────────────────────────────────────────────────
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []  # system 제외
    if "model_name" not in st.session_state:
        st.session_state.model_name = AVAILABLE_MODELS[0]
    if "log_history" not in st.session_state:
        st.session_state.log_history = []
    if "enable_logging" not in st.session_state:
        st.session_state.enable_logging = False
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"sess-{time.time()}"

initialize_session_state()

# ─────────────────────────────────────────────────────────────
# 3) 사이드바 UI
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("설정 및 도구")

    # API Key
    st.session_state.api_key = None
    try:
        if "GEMINI_API_KEY" in st.secrets:
            st.session_state.api_key = st.secrets["GEMINI_API_KEY"]
            st.success("API 키 로드 완료 (st.secrets)")
    except Exception:
        pass

    if st.session_state.api_key is None:
        st.session_state.api_key = st.text_input(
            "Gemini API Key를 입력하세요:",
            type="password",
            placeholder="API 키가 없으면 기능을 사용할 수 없습니다.",
        )

    # 모델 선택
    st.session_state.model_name = st.selectbox(
        "사용할 Gemini 모델 선택:",
        options=AVAILABLE_MODELS,
        index=0,
    )

    # 로깅 옵션
    st.session_state.enable_logging = st.checkbox(
        "대화 자동 CSV 기록 활성화", value=st.session_state.enable_logging
    )

    st.markdown("---")

    # 대화 초기화
    if st.button("대화 초기화", type="primary"):
        st.session_state.messages = []
        st.session_state.log_history = []
        st.rerun()

    st.subheader("세션 정보 및 로그")
    st.write(f"**현재 모델:** {st.session_state.model_name}")
    st.write(f"**세션 ID:** {st.session_state.get('session_id')}")

    # 로그 다운로드
    if st.session_state.log_history:
        csv_buffer = io.StringIO()
        fieldnames = ["timestamp", "role", "content"]
        writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
        writer.writeheader()
        for item in st.session_state.log_history:
            writer.writerow(item)
        st.download_button(
            label="대화 로그 (CSV) 다운로드",
            data=csv_buffer.getvalue(),
            file_name="chat_log.csv",
            mime="text/csv",
        )
    else:
        st.info("아직 저장된 대화가 없습니다.")

# ─────────────────────────────────────────────────────────────
# 4) 메시지 기록 함수
# ─────────────────────────────────────────────────────────────
def log_message(role, content):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    if st.session_state.enable_logging:
        st.session_state.log_history.append(
            {"timestamp": timestamp, "role": role, "content": content}
        )

    st.session_state.messages.append(
        {"role": role, "parts": [{"text": content}]}
    )

# ─────────────────────────────────────────────────────────────
# 5) API용 히스토리 변환 (system 제거 + 역할 변환)
# ─────────────────────────────────────────────────────────────
def build_api_history():
    api_history = []
    for m in st.session_state.messages:
        role = "model" if m["role"] == "assistant" else "user"
        api_history.append({"role": role, "parts": m["parts"]})
    return api_history

# ─────────────────────────────────────────────────────────────
# 6) 재시도 포함 API 호출
# ─────────────────────────────────────────────────────────────
def retry_api_call(client, model, contents, **kwargs):
    try:
        return client.models.generate_content(
            model=model,
            contents=contents,
            **kwargs
        )
    except Exception as e:
        st.error(f"API 오류: {e}")
        return None

# ─────────────────────────────────────────────────────────────
# 7) 메인 채팅 루프
# ─────────────────────────────────────────────────────────────
def get_assistant_response(user_prompt: str):
    if not st.session_state.api_key:
        st.error("Gemini API 키를 입력해야 합니다.")
        return

    client = genai.Client(api_key=st.session_state.api_key)

    history = build_api_history()
    if len(history) > 6:
        history = history[-6:]

    with st.spinner(f"({st.session_state.model_name}) 모델이 답변을 생성하는 중..."):
        response = retry_api_call(
            client,
            st.session_state.model_name,
            history,
            system_instruction=SYSTEM_INSTRUCTION
        )

    if not response:
        st.error("응답 생성 실패")
        return

    answer = response.text
    with st.chat_message("assistant"):
        st.markdown(answer)

    log_message("assistant", answer)

# ─────────────────────────────────────────────────────────────
# 8) UI 출력
# ─────────────────────────────────────────────────────────────
st.title("🌟 AI 고객 응대 센터 챗봇")
st.caption("장비 대여 문의 및 불편 사항을 접수해 드립니다.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["parts"][0]["text"])

user_prompt = st.chat_input("문의 사항을 입력해 주세요.")
if user_prompt:
    with st.chat_message("user"):
        st.markdown(user_prompt)
    log_message("user", user_prompt)
    get_assistant_response(user_prompt)
