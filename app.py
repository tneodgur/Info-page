import streamlit as st
import time
import csv
import io
import json
from google import genai
from google.genai.errors import APIError

def build_api_history():
    api_history = []
    for m in st.session_state.messages:
        if m["role"] == "system":
            continue  # system은 빼기
        role = "model" if m["role"] == "assistant" else "user"
        api_history.append({"role": role, "parts": m["parts"]})
    return api_history
    
# ─────────────────────────────────────────────────────────────
# 0) Streamlit 기본 설정 (가장 먼저 호출 권장)
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

SYSTEM_INSTRUCTION_PARTS = [
    {
        "text": """
당신은 고객의 장비 대여 가능 여부 문의 및 불편 사항을 접수하는 전문 고객 응대 챗봇입니다. 
사용자의 상황에 깊이 공감하고, 매우 정중하며 친절한 말투로 응대해야 합니다.

응대 규칙:
1. 사용자의 불편 사항이나 장비 대여 문의에 대해 공감하고 정중하게 응답하십시오.
2. 답변 시, 사용자의 불편 사항 또는 장비 대여 요청 내용을 '무엇이', '언제', '어디서', '어떻게'에 맞춰 구체적으로 요약하고, 이를 고객 응대 담당자에게 전달하여 신속히 처리하겠다는 취지로 안내해야 합니다.
3. 모든 응답의 마지막에는 담당자 확인 후 회신을 위해 반드시 사용자의 이메일 주소를 정중하게 요청하십시오.
4. 만일 사용자가 이메일 주소 제공을 명시적으로 거부할 경우, 다음 문구를 그대로 사용하여 정중히 안내하십시오: "죄송하지만, 연락처 정보를 받지 못하여 담당자의 검토 내용을 받으실 수 없어요."
"""
    }
]

# ─────────────────────────────────────────────────────────────
# 2) 세션 상태 초기화
# ─────────────────────────────────────────────────────────────
def initialize_session_state():
    if "messages" not in st.session_state:
        # API에 보낼 대화 히스토리: 첫 메시지는 시스템 프롬프트
        st.session_state.messages = [
            {"role": "system", "parts": SYSTEM_INSTRUCTION_PARTS}
        ]
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
# 3) 상단 UI
# ─────────────────────────────────────────────────────────────
st.title("🌟 AI 고객 응대 센터 챗봇")
st.caption("장비 대여 문의 및 불편 사항을 접수해 드립니다.")

# ─────────────────────────────────────────────────────────────
# 4) 사이드바 (설정/도구)
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("설정 및 도구")

    # 4-1) API Key
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

    # 4-2) 모델 선택
    st.session_state.model_name = st.selectbox(
        "사용할 Gemini 모델 선택:",
        options=AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(st.session_state.model_name)
        if st.session_state.model_name in AVAILABLE_MODELS
        else 0,
        help="gemini-2.0-flash가 기본 응대 모델로 적합합니다.",
    )

    # 4-3) 자동 CSV 로깅
    st.session_state.enable_logging = st.checkbox(
        "대화 자동 CSV 기록 활성화", value=st.session_state.enable_logging
    )

    st.markdown("---")

    # 4-4) 대화 초기화
    if st.button("대화 초기화", type="primary"):
        # 시스템 메시지는 유지하고, 사용자/어시스턴트 히스토리만 정리
        st.session_state.messages = [
            {"role": "system", "parts": SYSTEM_INSTRUCTION_PARTS}
        ]
        st.session_state.log_history = []
        st.rerun()

    # 4-5) 세션 정보 & 로그 다운로드
    st.subheader("세션 정보 및 로그")
    st.write(f"**현재 모델:** {st.session_state.model_name}")
    st.write(f"**세션 ID:** {st.session_state.get('session_id', '미할당')}")

    if st.session_state.log_history:
        csv_buffer = io.StringIO()
        fieldnames = ["timestamp", "role", "content"]
        writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
        writer.writeheader()
        for item in st.session_state.log_history:
            content_text = item["content"]
            writer.writerow(
                {
                    "timestamp": item["timestamp"],
                    "role": item["role"],
                    "content": content_text.replace("\n", " "),
                }
            )
        st.download_button(
            label="대화 로그 (CSV) 다운로드",
            data=csv_buffer.getvalue(),
            file_name=f"chatbot_log_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    else:
        st.warning("다운로드할 대화 기록이 없습니다.")

# ─────────────────────────────────────────────────────────────
# 5) 공용 함수
# ─────────────────────────────────────────────────────────────
def retry_api_call(client, model, contents, max_retries=3, initial_delay=1):
    """
    429 (속도 제한/할당 초과)에서만 재시도. 지수 백오프.
    """
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(model=model, contents=contents)
        except APIError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                st.warning(
                    f"API 호출 실패 (시도 {attempt + 1}/{max_retries}): 속도 제한 (429). "
                    f"{delay:.1f}초 후 재시도합니다."
                )
                time.sleep(delay)
                delay *= 2
            else:
                st.error(f"API 호출 중 복구 불가능한 오류 발생: {e}")
                return None
        except Exception as e:
            st.error(f"예상치 못한 오류 발생: {e}")
            return None
    st.error(f"최대 {max_retries}회 재시도 후에도 API 호출에 실패했습니다.")
    return None


def log_message(role, content):
    """
    화면 표시/CSV 로깅용 기록.
    - 'system'은 이미 초기 메시지에 존재하므로 messages에는 추가하지 않고,
      CSV 로깅에만 남깁니다.
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # CSV 로깅
    if st.session_state.enable_logging:
        st.session_state.log_history.append(
            {"timestamp": timestamp, "role": role, "content": content}
        )

    # 대화 히스토리 (API 전송용)
    if role in ("user", "assistant"):
        st.session_state.messages.append({"role": role, "parts": [{"text": content}]})


# ─────────────────────────────────────────────────────────────
# 6) 메인 채팅 루프
# ─────────────────────────────────────────────────────────────
def main_chat_loop():
    def get_assistant_response(user_prompt: str):
        if not st.session_state.api_key:
            st.error("Gemini API 키를 입력해야 챗봇을 사용할 수 있습니다.")
            return

        try:
            client = genai.Client(api_key=st.session_state.api_key)
        except Exception as e:
            st.error(f"API 클라이언트 초기화 오류: {e}")
            return

        # 최근 3턴(6메시지)만 유지 + 시스템 프롬프트
        history_to_send = build_api_history()

# 최근 6개 메시지만 유지
if len(history_to_send) > 6:
    history_to_send = history_to_send[-6:]

        with st.spinner(f"({st.session_state.model_name}) 모델이 답변을 생성하는 중..."):
            response = retry_api_call(
    client=client,
    model=st.session_state.model_name,
    contents=history_to_send,
    system_instruction=SYSTEM_INSTRUCTION_PARTS[0]["text"]
)

            )

        if response and getattr(response, "candidates", None):
            assistant_response = response.text
            with st.chat_message("assistant"):
                st.markdown(assistant_response)
            log_message("assistant", assistant_response)
        elif response is not None:
            st.error("응답을 생성할 수 없습니다. 안전 정책에 의해 차단되었을 수 있습니다.")
        else:
            st.error("API 응답을 받지 못했습니다. 잠시 후 다시 시도해 주세요.")

    # 이전 대화 렌더링 (system 제외)
    for msg in st.session_state.messages:
        if msg["role"] in ("user", "assistant"):
            with st.chat_message(msg["role"]):
                text_content = msg["parts"][0]["text"]
                st.markdown(text_content)

    # 입력창
    user_prompt = st.chat_input("문의 사항이나 불편 사항을 입력해 주세요.")
    if user_prompt:
        # 화면/CSV 기록
        with st.chat_message("user"):
            st.markdown(user_prompt)
        log_message("user", user_prompt)

        # 응답 생성
        get_assistant_response(user_prompt)


# ─────────────────────────────────────────────────────────────
# 7) 앱 실행
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 최초 실행 시, 시스템 프롬프트를 CSV에만 기록(대화 히스토리는 이미 존재)
    if st.session_state.enable_logging:
        log_message("system", SYSTEM_INSTRUCTION_PARTS[0]["text"])

    main_chat_loop()
