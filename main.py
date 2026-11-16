"""
================================================================================
 test2.py — 육면체 크기+색상 감지 + Modbus 전송 (D0=크기, D1=색상)
================================================================================
 물체 규격 (최소 변 = 35mm 고정):
   소(35×35×35)  → 정육면체, 종횡비 ≈1.0  → size code 1
   중(35×35×55)  → 종횡비 ≈1.57          → size code 2
   대(35×35×85)  → 종횡비 ≈2.43          → size code 3

 HR[0] (D0) = 크기 코드 (0=미감지, 1=소, 2=중, 3=대)
 HR[1] (D1) = 색상 코드 (0=없음, 1=빨강, 2=초록, 3=파랑, 4=노랑, 5=흰색)
================================================================================
"""

import cv2
import numpy as np
import threading
import asyncio
import logging
from pymodbus.server import ModbusSerialServer
from pymodbus.simulator import SimData, SimDevice, DataType


# ═════════════════════════════════════════════════════════════════════════════
#  설정 상수
# ═════════════════════════════════════════════════════════════════════════════

# ── Modbus 통신 설정 ──
SERIAL_PORT      = "COM5"
BAUDRATE         = 9600
BYTESIZE        = 8
PARITY          = "N"
STOPBITS         = 1

# ── Modbus 슬레이브 메모리 설정 ──
SLAVE_ID         = 6
HR_ADDRESS       = 0
HR_COUNT         = 16            # PLC가 16개를 연속 읽기하므로 16개 생성 (사용은 HR[0]=크기, HR[1]=색상 2개만)

# ── 카메라 설정 ──
CAMERA_INDEX      = None          # None=자동 탐색, 0/1=강제 지정

# ── 카메라 영상 조정 (너무 밝은/어두운 환경 보정) ──
#     OpenCV CAP_PROP 범위는 0~255 (0~1 아님)
#     brightness: 밝기 (기본 64 → 낮추어 과노출 방지, 너무 낮으면 어두워지니 주의)
#     saturation : 채도 (기본 4 → 높여 색상 대비 강화)
#     auto_exposure: -1(수동) → 1(자동노출) 또는 0.25(짧은 노출)
#     gain: 자동 이득 조정 (1=자동, 0=수동)
CAMERA_BRIGHTNESS  = 50             # 밝기 (0~255, 기본 64 → 30으로 낮춤)
CAMERA_SATURATION   = 130            # 채도 (0~255, 기본 4 → 128로 높임)
CAMERA_AUTO_EXPOSURE = 0.25           # 자동노출 (0.25 = 짧은 노출시간, 과노출 방지)
CAMERA_GAIN         = 0              # 이득 (0=수동, 자동이득 끄어 밝기 제어)

# ── 색상 감지 임계값 ──
MIN_AREA_RATIO    = 0.003
APPROX_EPS_RATIO  = 0.025
MIN_SOLIDITY      = 0.90
MIN_FILL_RATIO     = 0.45
MIN_ASPECT        = 0.30
MAX_ASPECT        = 2.50

# ── 크기 판별 임계값 (종횡비 기준) ──
#     종횡비 = max(w, h) / min(w, h)  (회전 무관)
#     소(35×35×35): 종횡비 ≈1.0  → < 1.3
#     중(35×35×55): 종횡비 ≈1.57 → 1.3 ~ 1.9
#     대(35×35×85): 종횡비 ≈2.43 → ≥ 1.9
SIZE_SMALL_MAX     = 1.30          # 종횡비 < 이 값 → 소
SIZE_MEDIUM_MAX    =1.90          # 종횡비 < 이 값 → 중, 이상 → 대


# ═════════════════════════════════════════════════════════════════════════════
#  로깅 설정
# ═════════════════════════════════════════════════════════════════════════════

logging.getLogger("pymodbus").setLevel(logging.WARNING)

log = logging.getLogger("Vision-Modbus")
log.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter(
    "%(asctime)s │ %(message)s",
    datefmt="%H:%M:%S",
))
log.addHandler(_handler)


# ═════════════════════════════════════════════════════════════════════════════
#  매핑 테이블
# ═════════════════════════════════════════════════════════════════════════════

FUNC_CODE_DESC = {
    1: "코일 읽기 (FC01)",
    2: "입력 코일 읽기 (FC02)",
    3: "홀딩 레지스터 읽기 (FC03)",
    4: "입력 레지스터 읽기 (FC04)",
    5: "코일 쓰기 (FC05)",
    6: "홀딩 레지스터 1개 쓰기 (FC06)",
    15: "코일 다수 쓰기 (FC0F)",
    16: "홀딩 레지스터 다수 쓰기 (FC10)",
}

COLOR_NAMES = {
    0: "없음",
    1: "빨강",
    2: "초록",
    3: "파랑",
    4: "노랑",
    5: "흰색",
}

SIZE_NAMES = {
    0: "미감지",
    1: "소",
    2: "중",
    3: "대",
}


# ═════════════════════════════════════════════════════════════════════════════
#  HSV 색상 범위
# ═════════════════════════════════════════════════════════════════════════════

COLOR_RANGES = {
    "WHITE": {
        "code": 5,
        "ranges": [
            (np.array([0,   0, 180]), np.array([179,  40, 255])),
        ]
    },
    "RED": {
        "code": 1,
        "ranges": [
            (np.array([0,   100,  80]), np.array([8,   255, 255])),
            (np.array([172, 100,  80]), np.array([179, 255, 255])),
        ]
    },
    "GREEN": {
        "code": 2,
        "ranges": [
            (np.array([38,  80, 70]),  np.array([82,  255, 255])),
        ]
    },
    "BLUE": {
        "code": 3,
        "ranges": [
            (np.array([95,  100, 70]),  np.array([130, 255, 255])),
        ]
    },
    "YELLOW": {
        "code": 4,
        "ranges": [
            (np.array([22, 100, 80]),  np.array([33,  255, 255])),
        ]
    },
}


# ═════════════════════════════════════════════════════════════════════════════
#  유틸리티 함수
# ═════════════════════════════════════════════════════════════════════════════

def format_hex(data: bytes) -> str:
    """bytes → HEX 문자열"""
    return " ".join(f"{b:02X}" for b in data)


def trace_packet(sent: bool, data: bytes) -> bytes:
    """Modbus 패킷 로그 (현재 주석 처리 — 통신 로그 출력 안 함)

    통신 로그가 필요하면 아래 return data 위의 주석을 해제하세요.
    """
    # ── 주석 처리: 통신 패킷 로그 비활성화 ──
    # direction = "▶ 송신" if sent else "◀ 수신"
    # hex_str = format_hex(data)
    #
    # if len(data) >= 2:
    #     dev_id = data[0]
    #     func_code = data[1]
    #     func_desc = FUNC_CODE_DESC.get(func_code, f"FC{func_code:02X}")
    #
    #     if not sent and func_code in (3, 4) and len(data) >= 6:
    #         addr = (data[2] <<8) | data[3]
    #         count = (data[4] <<8) | data[5]
    #         detail = f"국번={dev_id} │ {func_desc} │ 주소={addr} │ 개수={count}"
    #     elif sent and func_code in (3,4) and len(data) >= 3:
    #         byte_count = data[2]
    #         regs = []
    #         for i in range(3, min(3 + byte_count, len(data)),2):
    #             if i + 1 < len(data):
    #                 regs.append((data[i] <<8) | data[i + 1])
    #         reg_desc = ", ".join(f"{r}" for r in regs)
    #         desc_parts = []
    #         for j, r in enumerate(regs):
    #             if j == 0:
    #                 desc_parts.append(SIZE_NAMES.get(r, f"{r}"))
    #             elif j == 1:
    #                 desc_parts.append(COLOR_NAMES.get(r, f"{r}"))
    #             else:
    #                 desc_parts.append(f"{r}")
    #         detail = f"국번={dev_id} │ {func_desc} │ 값=[{reg_desc}] → [{', '.join(desc_parts)}]"
    #     elif not sent and func_code == 6 and len(data) >= 6:
    #         addr = (data[2] <<8) | data[3]
    #         val = (data[4] <<8) | data[5]
    #         detail = f"국번={dev_id} │ {func_desc} │ 주소={addr} │ 값={val}"
    #     else:
    #         detail = f"국번={dev_id} │ {func_desc}"
    # else:
    #     detail = "(패킷 너무 짧음)"
    # log.info(f"{direction} │ {detail}")
    # log.info(f"         HEX: {hex_str}")
    return data


def trace_connect(connected: bool) -> None:
    """RS-485 연결/동결 로그 (현재 주석 처리)"""
    # status = "연결됨" if connected else "동결됨"
    # log.info(f"🔌 RS485 {status}")
    pass


# ═════════════════════════════════════════════════════════════════════════════
#  육면체 형태 검증
# ═════════════════════════════════════════════════════════════════════════════

def is_hexahedron_shape(contour):
    """윤곽선이 직선 육면체 형태인지 엄격 판별 (5단계)"""
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, APPROX_EPS_RATIO * peri, True)

    if not (4 <= len(approx) <= 6):
        return False, None
    if not cv2.isContourConvex(approx):
        return False, None

    area = cv2.contourArea(contour)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    if hull_area == 0 or (area / hull_area) < MIN_SOLIDITY:
        return False, None

    x, y, w, h = cv2.boundingRect(approx)
    if w * h == 0 or (area / (w * h)) < MIN_FILL_RATIO:
        return False, None

    aspect = w / h if h > 0 else 0
    if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
        return False, None

    return True, approx


# ═════════════════════════════════════════════════════════════════════════════
#  크기 판별 — 종횡비로 소/중/대 분류
# ═════════════════════════════════════════════════════════════════════════════

def classify_size(approx):
    """바운딩 박스 종횡비로 크기 분류

    물체 규격 (최소 변 = 35mm 고정):
       소(35×35×35): 종횡비 ≈1.0  → code 1
       중(35×35×55): 종횡비 ≈1.57 → code 2
       대(35×35×85): 종횡비 ≈2.43 → code3

    :param approx: cv2.approxPolyDP 결과 (꼭짓점 배열)
    :return: (size_code, size_name, w, h)
    """
    x, y, w, h = cv2.boundingRect(approx)
    # 종횡비 = 긴 변 / 짧은 변 (회전 무관, 항상 ≥1)
    ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1.0

    if ratio < SIZE_SMALL_MAX:
        return 1, "소", w, h
    elif ratio < SIZE_MEDIUM_MAX:
        return 2, "중", w, h
    else:
        return 3, "대", w, h


# ═════════════════════════════════════════════════════════════════════════════
#  1. Modbus 메모리 초기화 — HR[0]=크기, HR[1]=색상
# ═════════════════════════════════════════════════════════════════════════════

hr_data = SimData(address=HR_ADDRESS, values=[0] * HR_COUNT, datatype=DataType.REGISTERS)
device = SimDevice(id=SLAVE_ID, simdata=[hr_data])

# 국번 5번 더미 장치 (버스에 물린 타 장비 에러 방지)
hr_data_dummy5 = SimData(address=0, values=[0] * 10, datatype=DataType.REGISTERS)
device_dummy5 = SimDevice(id=5, simdata=[hr_data_dummy5])

# 국번 0번 더미 장치 (브로드캐스트 요청 에러 방지)
hr_data_dummy0 = SimData(address=0, values=[0] * 10, datatype=DataType.REGISTERS)
device_dummy0 = SimDevice(id=0, simdata=[hr_data_dummy0])


# ═════════════════════════════════════════════════════════════════════════════
#  2. 백그라운드 스레드 — Modbus RTU Slave 서버
# ═════════════════════════════════════════════════════════════════════════════

server = None
server_loop = None

async def async_server_main():
    global server, server_loop
    server_loop = asyncio.get_running_loop()
    server = ModbusSerialServer(
        context=[device, device_dummy5, device_dummy0],
        port=SERIAL_PORT,
        baudrate=BAUDRATE,
        bytesize=BYTESIZE,
        parity=PARITY,
        stopbits=STOPBITS,
        trace_packet=trace_packet,
        trace_pdu=None,
        trace_connect=trace_connect,
        ignore_missing_devices=True,
    )
    log.info(f"Modbus RTU Slave 서버 시작 ({SERIAL_PORT}, {BAUDRATE}bps, 국번 {SLAVE_ID})")
    await server.serve_forever()

def run_modbus_server():
    asyncio.run(async_server_main())

server_thread = threading.Thread(target=run_modbus_server, daemon=True)
server_thread.start()


# ═════════════════════════════════════════════════════════════════════════════
#  3. 카메라 연결 (자동 탐색)
# ═════════════════════════════════════════════════════════════════════════════

def find_camera():
    """사용 가능한 카메라 자동 탐색 (외장 USB 우선)

    카메라 연결 후 밝기/채도/노출 조정:
      brightness → CAM_BRIGHTNESS (과노출 방지, 0~255)
      saturation → CAM_SATURATION (색상 선명도, 0~255)
      auto_exposure → CAM_AUTO_EXPOSURE (짧은 노출, 과노출 방지)
      gain → CAM_GAIN (수동 이득)
    """
    if CAMERA_INDEX is not None:

        cap = cv2.VideoCapture(CAMERA_INDEX)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BRIGHTNESS, CAMERA_BRIGHTNESS)
            cap.set(cv2.CAP_PROP_SATURATION, CAMERA_SATURATION)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, CAMERA_AUTO_EXPOSURE)
            cap.set(cv2.CAP_PROP_GAIN, CAMERA_GAIN)
            log.info(f"카메라 연결: 인덱스 {CAMERA_INDEX} (지정) | "
                    f"brightness={CAMERA_BRIGHTNESS}, saturation={CAMERA_SATURATION}")
            return cap
        cap.release()
        log.warning(f"지정 인덱스 {CAMERA_INDEX} 열 수 없음. 자동 탐색 전환.")

    for idx in range(10):
        if idx == 0:
            continue
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cap.set(cv2.CAP_PROP_BRIGHTNESS, CAMERA_BRIGHTNESS)
                cap.set(cv2.CAP_PROP_SATURATION, CAMERA_SATURATION)
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, CAMERA_AUTO_EXPOSURE)
                cap.set(cv2.CAP_PROP_GAIN, CAMERA_GAIN)
                log.info(f"카메라 연결: 인덱스 {idx} (자동 탐색) | "
                        f"brightness={CAMERA_BRIGHTNESS}, saturation={CAMERA_SATURATION}")
                return cap
            cap.release()
        else:
            cap.release()
    return None

cap = find_camera()
if cap is None:
    log.error("사용 가능한 카메라를 찾을 수 없습니다. USB 카메라 연결 확인 필요.")
    raise RuntimeError("카메라 없음")


# ═════════════════════════════════════════════════════════════════════════════
#  4. 메인 루프 — 가장 큰 육면체 1개의 크기+색상을 HR[0], HR[1]에 전송
# ═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

last_sent = None  # (size_code, color_code) 튜플 — 중복 전송 방지

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            log.warning("카메라를 읽을 수 없습니다. "
                        "다른 프로그램(Zoom, Teams 등)이 카메라를 점유 중인지 확인하세요.")
            break

        h_frame, w_frame = frame.shape[:2]
        frame_area = h_frame * w_frame

        # 노이즈 제거 + HSV 변환
        blurred = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # 색상별 육면체 감지 — 가장 큰 객체 1개만 선택
        min_area = int(frame_area * MIN_AREA_RATIO)
        best_area = 0
        best_name = "NONE"
        best_code = 0
        best_size_code = 0
        best_size_name = "미감지"
        best_approx = None

        for color_name, info in COLOR_RANGES.items():
            mask = None
            for lower, upper in info["ranges"]:
                current_mask = cv2.inRange(hsv, lower, upper)
                mask = current_mask if mask is None else cv2.bitwise_or(mask, current_mask)

            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > min_area:
                    is_cube, approx = is_hexahedron_shape(cnt)
                    if is_cube and area > best_area:
                        best_area = area
                        best_name = color_name
                        best_code = info["code"]
                        best_approx = approx
                        # 크기 분류 (종횡비 기준)
                        best_size_code, best_size_name, _, _ = classify_size(approx)

        # 감지 결과 화면 표시
        if best_approx is not None:
            cv2.drawContours(frame, [best_approx], -1, (0, 255, 0), 3)
            for point in best_approx:
                px, py = point[0]
                cv2.circle(frame, (px, py), 5, (0, 255, 255), -1)
            x, y, w, h = cv2.boundingRect(best_approx)
            label = f"{best_size_name} {best_name} CUBE (D0={best_size_code}, D1={best_code})"
            cv2.putText(frame, label, (x, max(y - 12, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        # Modbus 전송 — 크기 or 색상이 변경될 때만
        current = (best_size_code, best_code)
        if (server is not None and server_loop is not None
                and current != last_sent):
            # HR[0] = 크기 코드, HR[1] = 색상 코드
            asyncio.run_coroutine_threadsafe(
                server.async_setValues(SLAVE_ID, 3, HR_ADDRESS,
                                        [best_size_code, best_code]),
                server_loop
            )
            size_desc = SIZE_NAMES.get(best_size_code, "알 수 없음")
            color_desc = COLOR_NAMES.get(best_code, "알 수 없음")
            log.info(
                f"📦 Modbus 전송 │ HR[{HR_ADDRESS}]={best_size_code} ({size_desc}), "
                f"HR[{HR_ADDRESS + 1}]={best_code} ({color_desc}) "
                f"← {best_size_name} {best_name}"
            )
            last_sent = current

        cv2.putText(frame,
                    f"D0={best_size_code}({best_size_name}) | D1={best_code}({best_name})",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Workpiece Vision Slave", frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    log.info("프로그램 종료")
