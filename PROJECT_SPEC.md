# PROJECT SPEC.md

# NỀN TẢNG ĐIỀU PHỐI TRẠM SẠC EV BẰNG HỌC TĂNG CƯỜNG

## AI VIBE CODING & IMPLEMENTATION SPECIFICATION

**Version:** 1.0  
**Status:** Implementation-ready specification  
**Primary language:** Vietnamese  
**Project type:** Reinforcement Learning + Traffic Micro-simulation + Web GIS Dashboard

---

# 1. MỤC ĐÍCH CỦA TÀI LIỆU

Tài liệu này là specification chính thức để AI coding agent, đặc biệt là Claude Code, sử dụng làm nguồn tham chiếu khi xây dựng toàn bộ dự án.

AI phải coi tài liệu này là **source of truth** cho:

- kiến trúc hệ thống;
- simulation;
- MDP;
- observation space;
- action space;
- reward;
- battery dynamics;
- charging station;
- traffic;
- RL training;
- baseline;
- evaluation;
- backend API;
- WebSocket;
- frontend;
- testing.

AI KHÔNG được tự ý thay đổi các thành phần cốt lõi của specification nếu chưa được người dùng phê duyệt.

Nếu specification có điểm chưa rõ hoặc có hai cách triển khai hợp lý, AI phải:

1. phát hiện ambiguity;
2. giải thích ngắn gọn;
3. đề xuất phương án;
4. chờ người dùng quyết định nếu thay đổi đó ảnh hưởng đến kiến trúc, MDP hoặc kết quả nghiên cứu.

Không tự ý chọn một phương án rồi âm thầm thay đổi specification.

---

# 2. VAI TRÒ CỦA AI CODING AGENT

AI đóng vai trò:

- Senior Systems Engineer;
- Reinforcement Learning Engineer;
- Python Engineer;
- Backend Engineer;
- Full-Stack Developer.

AI phải ưu tiên:

1. tính đúng đắn;
2. tính tái lập của thí nghiệm;
3. kiến trúc đơn giản;
4. khả năng kiểm thử;
5. hiệu năng;
6. khả năng chạy trên máy tính cá nhân;
7. code dễ đọc và dễ bảo trì.

Ngôn ngữ giao tiếp với người dùng: **Tiếng Việt**.

Tên biến, class, function và code: **English**.

Không tạo code chỉ để "trông hiện đại".

Không thêm công nghệ chỉ vì công nghệ đó phổ biến.

---

# 3. BÀI TOÁN CẦN GIẢI QUYẾT

## 3.1. Problem

Hệ thống mô phỏng tình huống nhiều xe điện đồng thời cần sạc.

Nếu các xe đều lựa chọn trạm sạc gần nhất hoặc cùng một trạm đang có vẻ thuận lợi, có thể xảy ra hiện tượng:

**Thundering Herd / Crowd Effect**

Ví dụ:

```text
EV 1 ─┐
EV 2 ─┤
EV 3 ─┤──> Charging Station A
EV 4 ─┤
EV 5 ─┘

Station A:
Queue quá dài
↓
Waiting time tăng
↓
Total system cost tăng
```

Mục tiêu của hệ thống là sử dụng Reinforcement Learning để phân phối EV giữa các charging station một cách hợp lý.

---

# 4. MỤC TIÊU TỐI ƯU

Mục tiêu chính:

> Giảm tổng chi phí thời gian của hệ thống, bao gồm thời gian di chuyển và thời gian chờ tại trạm sạc.

Định nghĩa:

```text
Total Cost =
    w1 * Total Travel Time
    + w2 * Total Waiting Time
```

RL phải học cách lựa chọn charging station sao cho Total Cost giảm.

Không đặt mục tiêu đơn thuần là:

```text
distance đến station nhỏ nhất
```

vì đó chính là baseline heuristic mà RL cần được so sánh.

---

# 5. PHẠM VI MÔ PHỎNG

Đây là micro-simulation.

Không xây dựng traffic simulator quy mô lớn.

## 5.1. Quy mô mặc định

```text
Traffic nodes:       20
Charging stations:    5
Electric vehicles:   50
```

Có thể cấu hình tối đa:

```text
Traffic nodes:       30
Charging stations:   10
Electric vehicles:   100
```

Các giá trị phải được đặt trong configuration thay vì hard-code rải rác trong code.

---

# 6. TECHNOLOGY STACK

## 6.1. Backend / Simulation / AI

Bắt buộc ưu tiên:

```text
Python 3.10+
NetworkX
NumPy
Gymnasium
Stable-Baselines3
PyTorch
FastAPI
Uvicorn
WebSocket
Pydantic
```

## 6.2. Database

```text
PostgreSQL
```

Database access:

```text
asyncpg
```

hoặc Supabase nếu cần.

Database không phải thành phần bắt buộc của RL core.

RL environment phải chạy độc lập mà không cần database.

---

# 7. FRONTEND

Bắt buộc:

```text
Next.js
App Router
TypeScript
Tailwind CSS
Lucide React
Leaflet
React-Leaflet
```

Không sử dụng Google Maps API.

Không yêu cầu API trả phí cho chức năng bản đồ cơ bản.

Frontend phải được thiết kế để có thể chạy độc lập với RL training.

---

# 8. NGUYÊN TẮC KIẾN TRÚC

Kiến trúc tổng thể:

```text
                 ┌─────────────────────┐
                 │   Traffic Network   │
                 │     NetworkX        │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   EV Simulation     │
                 │ Vehicle / Traffic   │
                 │ Charging Station    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │   Gymnasium Env     │
                 │      EVEnv          │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │       DQN           │
                 │ Stable-Baselines3   │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Evaluation /        │
                 │ Baseline Comparison │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ FastAPI Backend     │
                 │ WebSocket           │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────┐
                 │ Next.js Dashboard   │
                 │ Leaflet GIS         │
                 └─────────────────────┘
```

## 8.1. Separation of concerns

Không được trộn:

```text
RL training
```

với:

```text
FastAPI
WebSocket
Next.js
```

RL phải có thể train bằng command line mà không cần chạy web server.

---

# 9. PROJECT DIRECTORY

Kiến trúc mục tiêu:

```text
project-root/
│
├── backend/
│   │
│   ├── ai_core/
│   │   ├── ev_env.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── models/
│   │
│   ├── simulation/
│   │   ├── network_graph.py
│   │   ├── traffic_logic.py
│   │   ├── vehicle.py
│   │   ├── charging_station.py
│   │   └── simulator.py
│   │
│   ├── baseline/
│   │   ├── nearest_station.py
│   │   ├── shortest_time.py
│   │   └── least_queue.py
│   │
│   ├── api/
│   │   ├── main.py
│   │   ├── websockets.py
│   │   └── routes/
│   │
│   ├── tests/
│   │
│   ├── config.py
│   └── requirements.txt
│
├── frontend/
│   │
│   ├── src/
│   │   ├── app/
│   │   ├── components/
│   │   │   ├── MapComponent.tsx
│   │   │   ├── DashboardCharts.tsx
│   │   │   └── ControlPanel.tsx
│   │   │
│   │   └── lib/
│   │       ├── websocket.ts
│   │       └── types.ts
│   │
│   └── package.json
│
├── data/
│
├── models/
│
├── results/
│
├── tests/
│
├── README.md
└── PROJECT_SPEC.md
```

Không tạo hàng chục folder khi chưa cần.

---

# 10. SIMULATION MODEL

## 10.1. Traffic graph

Sử dụng NetworkX.

Mặc định:

```text
20 nodes
```

Mỗi edge phải có tối thiểu:

```text
distance
traffic_weight
speed
travel_time
```

Trong đó:

```text
travel_time =
    distance / effective_speed
```

Traffic congestion có thể làm giảm effective speed.

---

# 11. NODE COORDINATES

Mỗi node phải có tọa độ.

Có thể sử dụng:

```text
x
y
```

cho simulation.

Nếu sử dụng Leaflet, cần có mapping sang:

```text
latitude
longitude
```

Không cần sử dụng bản đồ địa lý thật ở phiên bản đầu tiên.

Thiết kế data model phải cho phép thay thế bằng geographic coordinates sau này.

---

# 12. CHARGING STATION MODEL

Mỗi charging station có:

```text
station_id
node_id
capacity
num_chargers
queue
charging_vehicles
```

Ví dụ:

```text
Station A
capacity = 5
num_chargers = 2
queue = 3
charging = 2
```

Phân biệt rõ:

```text
queue
```

và:

```text
vehicles currently charging
```

Không gộp hai khái niệm này.

---

# 13. ELECTRIC VEHICLE MODEL

Mỗi EV tối thiểu có:

```text
vehicle_id
current_node
destination_node
battery_level
battery_capacity
speed
target_station
state
```

State có thể gồm:

```text
TRAVELING
WAITING
CHARGING
COMPLETED
FAILED
```

---

# 14. SIMULATION TIME

Simulation sử dụng discrete time.

```text
Δt = 1 simulated second
```

Mỗi tick:

```text
t = t + 1
```

Simulation phải có:

```text
simulation_time
```

Không dùng wall-clock time cho logic RL.

WebSocket có thể gửi dữ liệu mỗi 1 giây hoặc theo configurable interval.

---

# 15. BATTERY MODEL

Battery nằm trong:

```text
[0.0, 1.0]
```

Mỗi EV có battery consumption khi di chuyển.

Phiên bản đầu:

```text
battery -= energy_per_distance * distance_travelled
```

Giá trị phải được cấu hình.

Không hard-code battery consumption trong nhiều file.

Battery phải được clamp:

```text
0.0 <= battery_level <= 1.0
```

---

# 16. ROUTING

Đây là nguyên tắc quan trọng:

## RL KHÔNG chọn route trong phiên bản V1.

Action của RL chỉ là:

```text
charging_station_id
```

Sau khi RL chọn station:

```text
EV
 ↓
Selected Station
 ↓
NetworkX shortest path
 ↓
Vehicle movement
```

Route được tính bởi routing layer.

Mục đích:

- giảm action space;
- giảm complexity;
- dễ train;
- dễ debug;
- dễ giải thích kết quả;
- tách quyết định chiến lược khỏi routing.

Có thể mở rộng RL route selection ở phiên bản tương lai, nhưng không triển khai trong V1.

---

# 17. RL FORMULATION — MDP

## 17.1. Agent

Agent điều phối quyết định charging station cho một EV cần sạc.

V1 sử dụng:

```text
single-agent decision model
```

Không triển khai multi-agent RL trong V1.

---

# 18. OBSERVATION SPACE

Observation phải tập trung vào EV hiện tại và thông tin cần thiết để quyết định station.

Không đưa toàn bộ trạng thái của 100 EV vào observation nếu không cần thiết.

Observation gồm:

```text
1. EV current position
2. EV battery level
3. EV destination
4. Distance / travel-time tới mỗi charging station
5. Queue length tại mỗi station
6. Available charging capacity tại mỗi station
7. Traffic information liên quan tới các route
```

Dữ liệu phải được normalize về khoảng phù hợp, ưu tiên:

```text
[0, 1]
```

Observation space sử dụng:

```text
gymnasium.spaces.Box
```

---

# 19. ACTION SPACE

V1 bắt buộc:

```text
Discrete(number_of_charging_stations)
```

Ví dụ:

```text
5 stations
→ action_space = Discrete(5)
```

Mapping:

```text
action = 0 → Station 0
action = 1 → Station 1
...
```

Không dùng:

```text
station_id + route_id
```

trong V1.

Không dùng continuous action.

---

# 20. VALID ACTION

Một station được coi là valid nếu EV có khả năng tiếp cận station dựa trên:

```text
current battery
+
estimated energy required
```

Nếu station không thể tiếp cận an toàn:

```text
action invalid
```

Environment phải xử lý invalid action rõ ràng.

Không để vehicle âm battery.

---

# 21. REWARD FUNCTION

Reward cơ bản:

```text
R =
    -(w1 * travel_time
      + w2 * waiting_time)
```

Trong đó:

```text
travel_time = thời gian EV di chuyển tới station

waiting_time = thời gian EV phải chờ tại station
```

`w1` và `w2` phải là configuration parameters.

Ví dụ mặc định:

```text
w1 = 1.0
w2 = 1.0
```

Không được tự ý thay đổi trọng số trong quá trình implementation nếu chưa được phê duyệt.

---

# 22. CONGESTION PENALTY

Có thể bổ sung penalty cho tình trạng station quá tải hoặc traffic congestion.

Tuy nhiên:

**Không được để penalty lớn đến mức áp đảo toàn bộ reward nếu chưa kiểm tra scale.**

Mọi penalty phải có lý do và được cấu hình.

---

# 23. BATTERY FAILURE

Nếu EV hết pin giữa đường:

```text
battery_level <= 0
```

thì:

```text
reward = -1000
```

và episode có thể kết thúc với trạng thái failure.

Không được để EV tiếp tục di chuyển với battery âm.

Penalty `-1000` phải được cấu hình để dễ thay đổi và kiểm nghiệm.

---

# 24. STATION OVERLOAD

Không terminate toàn bộ episode chỉ vì một station đầy.

Nếu:

```text
queue > capacity
```

thì xử lý bằng:

```text
large negative penalty
```

hoặc invalid action tùy logic environment.

Simulation vẫn tiếp tục.

Lý do:

> Một station quá tải phải là trạng thái mà RL học cách tránh, không phải lý do để kết thúc toàn bộ simulation.

---

# 25. EPISODE TERMINATION

Episode có thể kết thúc khi:

```text
1. Tất cả EV hoàn thành nhiệm vụ
2. Simulation time vượt max_episode_steps
3. EV failure nghiêm trọng theo rule
```

Không terminate chỉ vì:

```text
một station đầy
```

---

# 26. RANDOMNESS & REPRODUCIBILITY

Tất cả simulation và training phải hỗ trợ:

```text
random seed
```

Ví dụ:

```text
seed = 42
```

Khi seed giống nhau:

```text
network
initial EV positions
traffic
battery
```

phải có khả năng tái lập ở mức phù hợp.

Không sử dụng random state không kiểm soát.

---

# 27. BASELINE METHODS

RL phải được so sánh với heuristic baselines.

V1 bắt buộc có tối thiểu:

## Baseline 1 — Nearest Station

Chọn station có khoảng cách ngắn nhất.

```text
argmin(distance)
```

## Baseline 2 — Shortest Travel Time

Chọn station có estimated travel time nhỏ nhất.

```text
argmin(travel_time)
```

## Baseline 3 — Least Queue

Chọn station có queue nhỏ nhất, có xét khả năng tiếp cận.

```text
argmin(queue)
```

Không được tuyên bố RL tốt hơn baseline trước khi thực nghiệm.

---

# 28. EVALUATION METRICS

Mỗi phương pháp phải được đánh giá bằng cùng scenario.

Metrics tối thiểu:

```text
Average Travel Time
Average Waiting Time
Total Travel Time
Total Waiting Time
Total System Cost
Average Queue Length
Maximum Queue Length
Station Utilization
Number of Failed EVs
Number of Overloaded Events
Episode Reward
```

---

# 29. FAIR COMPARISON

RL và baseline phải sử dụng:

```text
same network
same EV initial states
same battery states
same traffic conditions
same charging station configuration
same random seeds
```

Không được tạo scenario khác nhau rồi so sánh trực tiếp.

---

# 30. TRAIN / VALIDATION / TEST

Không đánh giá model trên đúng một scenario training.

Tối thiểu:

```text
Training scenarios
Validation scenarios
Test scenarios
```

Test scenario phải được giữ riêng.

Nếu dataset/simulation còn nhỏ, có thể sử dụng nhiều random seeds để đánh giá độ ổn định.

---

# 31. RL ALGORITHM

V1 bắt buộc sử dụng:

```text
DQN
```

với:

```text
Stable-Baselines3
MlpPolicy
```

Network phải nhỏ.

Ưu tiên:

```text
2–3 fully connected layers
```

với số neuron vừa phải.

Không sử dụng:

```text
Transformer
Large pretrained model
CNN
External foundation model
```

trừ khi specification được thay đổi.

PPO chỉ là hướng mở rộng hoặc experiment phụ.

---

# 32. TRAINING RESOURCE CONSTRAINT

Mục tiêu:

```text
Có thể train trên máy tính cá nhân.
```

Không thiết kế training yêu cầu GPU lớn.

Ưu tiên:

```text
small neural network
small observation space
small action space
reasonable episode length
```

Không tạo hàng triệu simulation objects không cần thiết.

Không chạy infinite loops.

Không tạo background worker nếu chưa cần.

---

# 33. SIMULATION PERFORMANCE

Ưu tiên:

```text
NumPy arrays
simple Python data structures
efficient NetworkX operations
```

Không tối ưu premature.

Nhưng phải tránh:

```text
nested loops không cần thiết
rebuild toàn bộ graph mỗi tick
recalculate toàn bộ route nếu route không thay đổi
gửi toàn bộ state qua WebSocket mỗi tick
```

---

# 34. WEBSOCKET DESIGN

Endpoint:

```text
/ws/simulation
```

WebSocket truyền simulation state tới frontend.

Không gửi toàn bộ static map mỗi tick.

Static data:

```text
network graph
station locations
node coordinates
```

được gửi một lần hoặc lấy qua REST endpoint.

Dynamic data:

```text
vehicle position
battery
vehicle state
station queue
charging state
traffic changes
```

chỉ gửi phần cần cập nhật.

---

# 35. WEBSOCKET MESSAGE

Ưu tiên message structure rõ ràng.

Ví dụ:

```json
{
  "type": "simulation_update",
  "timestamp": 42,
  "vehicles": [
    {
      "id": 1,
      "node": 7,
      "x": 0.42,
      "y": 0.71,
      "battery": 0.63,
      "state": "TRAVELING"
    }
  ],
  "stations": [
    {
      "id": 0,
      "queue": 3,
      "charging": 2
    }
  ]
}
```

Schema phải được định nghĩa bằng Pydantic ở backend và TypeScript interface ở frontend.

---

# 36. FASTAPI

FastAPI chịu trách nhiệm:

```text
REST API
WebSocket
simulation control
model inference
system status
```

FastAPI không chịu trách nhiệm train model dài hạn trong request thread.

Training phải chạy bằng command riêng.

---

# 37. DATABASE

PostgreSQL là persistence layer.

Database có thể lưu:

```text
simulation runs
experiment configurations
evaluation results
vehicle logs
station statistics
```

Không dùng database làm state store cho từng simulation tick nếu điều đó gây overhead không cần thiết.

Simulation state nên nằm trong memory trong runtime.

---

# 38. FRONTEND RESPONSIBILITIES

Frontend hiển thị:

## Map

```text
traffic network
charging stations
EV locations
EV movement
```

## Dashboard

```text
Total EVs
Active EVs
Average waiting time
Average travel time
Station utilization
Total system cost
```

## Control panel

Có thể gồm:

```text
Start
Pause
Reset
Scenario selection
Algorithm selection
Simulation speed
```

---

# 39. LEAFLET

Leaflet chỉ chịu trách nhiệm visualization.

Không để frontend tự tính:

```text
RL decision
routing
reward
battery dynamics
traffic simulation
```

Frontend chỉ hiển thị state nhận từ backend.

---

# 40. FRONTEND STATE MANAGEMENT

Không thêm Redux hoặc state management framework lớn nếu chưa cần.

Ưu tiên:

```text
React state
hooks
small utility modules
```

WebSocket client nằm trong:

```text
frontend/src/lib/websocket.ts
```

Types nằm trong:

```text
frontend/src/lib/types.ts
```

---

# 41. CODING RULES

## 41.1. No placeholders

Không sử dụng:

```text
TODO
FIXME
pass
...
NotImplementedError
```

trong implementation cuối cùng.

Nếu tính năng chưa được triển khai, phải nói rõ với người dùng thay vì giả vờ đã hoàn thành.

---

# 42. TYPE SAFETY

Python:

```text
Type hints
Dataclasses
Pydantic
```

Frontend:

```text
TypeScript
Interfaces
strict typing
```

Không sử dụng:

```text
Any
```

một cách tùy tiện.

---

# 43. ERROR HANDLING

Không nuốt exception.

Không sử dụng:

```python
except Exception:
    pass
```

Phải có error handling phù hợp và logging rõ ràng.

---

# 44. CONFIGURATION

Các parameter quan trọng phải tập trung.

Ví dụ:

```text
NUM_NODES
NUM_STATIONS
NUM_VEHICLES
MAX_EPISODE_STEPS
TIME_STEP
BATTERY_CONSUMPTION
STATION_CAPACITY
CHARGING_RATE
REWARD_TRAVEL_WEIGHT
REWARD_WAITING_WEIGHT
BATTERY_FAILURE_PENALTY
RANDOM_SEED
```

Không hard-code các giá trị này trong nhiều file.

---

# 45. TESTING REQUIREMENTS

Mỗi module quan trọng phải có test.

Tối thiểu:

```text
network graph creation
shortest path
battery update
station queue
vehicle movement
reward calculation
observation shape
action validity
episode termination
baseline decisions
```

Gym environment phải pass:

```text
reset()
step()
observation_space
action_space
```

và phải kiểm tra tương thích với Gymnasium/SB3.

---

# 46. DEVELOPMENT ORDER

Claude Code phải triển khai theo thứ tự sau.

Không nhảy thẳng vào frontend.

## PHASE 0 — Specification Validation

Kiểm tra:

```text
architecture
MDP
data models
simulation assumptions
```

Tạo một implementation plan.

Nếu có ambiguity quan trọng → báo người dùng.

Không code lớn ngay.

---

## PHASE 1 — Traffic Network

Implement:

```text
backend/simulation/network_graph.py
```

Yêu cầu:

```text
20 nodes
5 charging stations
node coordinates
edge distance
traffic weight
speed
travel time
shortest path
```

Viết test.

---

## PHASE 2 — Simulation Core

Implement:

```text
vehicle.py
charging_station.py
traffic_logic.py
simulator.py
```

Yêu cầu:

```text
vehicle movement
battery consumption
traffic
station queue
charging
simulation tick
```

Viết test.

---

## PHASE 3 — Gymnasium Environment

Implement:

```text
backend/ai_core/ev_env.py
```

Phải có:

```text
reset()
step()
observation_space
action_space
reward
termination
truncation
info
```

Test environment trước khi train.

---

## PHASE 4 — Baselines

Implement:

```text
nearest_station.py
shortest_time.py
least_queue.py
```

Chạy simulation.

Lưu metrics.

---

## PHASE 5 — DQN

Implement:

```text
backend/ai_core/train.py
```

Train DQN.

Save model vào:

```text
models/
```

Không train trong FastAPI.

---

## PHASE 6 — Evaluation

Implement:

```text
backend/ai_core/evaluate.py
```

Chạy:

```text
DQN
Nearest Station
Shortest Travel Time
Least Queue
```

trên cùng scenarios.

Lưu:

```text
results/
```

Sinh metrics có thể sử dụng cho báo cáo.

---

## PHASE 7 — FastAPI

Implement:

```text
backend/api/main.py
backend/api/websockets.py
```

REST + WebSocket.

Không phụ thuộc trực tiếp vào training process.

---

## PHASE 8 — Next.js Dashboard

Implement:

```text
MapComponent.tsx
DashboardCharts.tsx
ControlPanel.tsx
```

Frontend chỉ hiển thị backend state.

---

## PHASE 9 — Integration

Kiểm tra toàn bộ pipeline:

```text
Simulation
→ RL
→ FastAPI
→ WebSocket
→ Next.js
→ Leaflet
```

---

## PHASE 10 — Final Validation

Kiểm tra:

```text
tests
training
evaluation
frontend
API
WebSocket
resource usage
reproducibility
```

Sau đó mới coi V1 là hoàn thành.

---

# 47. DEFINITION OF DONE

Một phase chỉ được coi là hoàn thành khi:

```text
1. Code chạy được.
2. Không còn placeholder.
3. Có test phù hợp.
4. Không phá phase trước.
5. Import đúng architecture.
6. Có command để chạy.
7. Có output kiểm chứng.
```

Đối với RL:

```text
Environment chạy được
↓
Random/heuristic policy chạy được
↓
DQN train được
↓
Model save được
↓
Evaluation chạy được
↓
Metrics được lưu
```

Không được coi:

```text
"model đã train"
```

là hoàn thành nếu chưa có evaluation.

---

# 48. COMMAND STRUCTURE

Claude Code phải duy trì các command rõ ràng.

Ví dụ:

```bash
python -m backend.simulation.simulator
```

```bash
python -m backend.ai_core.train
```

```bash
python -m backend.ai_core.evaluate
```

```bash
uvicorn backend.api.main:app --reload
```

Frontend:

```bash
npm run dev
```

Nếu project thực tế có command khác, cập nhật README nhưng không phá cấu trúc module.

---

# 49. GIT RULES

Mỗi phase nên có commit riêng.

Ví dụ:

```text
feat: implement traffic network
feat: implement EV simulation
feat: implement Gymnasium environment
feat: add baseline policies
feat: add DQN training
feat: add evaluation pipeline
feat: add FastAPI websocket
feat: add Next.js dashboard
```

Không commit:

```text
.env
API keys
passwords
tokens
model secrets
```

---

# 50. CLAUDE CODE BEHAVIOR RULES

Claude Code phải tuân thủ:

## Rule 1

Không tự ý thay đổi MDP.

## Rule 2

Không tự ý đổi DQN sang PPO.

## Rule 3

Không tự ý biến action thành:

```text
station + route
```

## Rule 4

Không tự ý thêm multi-agent RL.

## Rule 5

Không tự ý thêm database dependency vào RL environment.

## Rule 6

Không tự ý thêm cloud service.

## Rule 7

Không tự ý thêm large AI model.

## Rule 8

Không code frontend trước khi simulation backend có state contract rõ ràng.

## Rule 9

Không tạo abstraction phức tạp nếu chưa cần.

## Rule 10

Không sửa nhiều file không liên quan đến task hiện tại.

---

# 51. FILE CHANGE DISCIPLINE

Khi được yêu cầu sửa một feature:

1. Xác định file liên quan.
2. Đọc code hiện tại.
3. Xác định dependency.
4. Sửa tối thiểu số file cần thiết.
5. Chạy test liên quan.
6. Kiểm tra regression.
7. Báo rõ file nào đã thay đổi.

Không rewrite toàn bộ project nếu chỉ cần sửa một function.

---

# 52. BEFORE CODING PROTOCOL

Trước mỗi task lớn, Claude Code phải trả lời ngắn:

```text
1. Tôi đang thay đổi thành phần nào?
2. Thành phần đó phụ thuộc vào gì?
3. Có ảnh hưởng MDP không?
4. Có ảnh hưởng API contract không?
5. Có cần sửa test không?
6. Có ambiguity nào không?
```

Nếu không có vấn đề:

```text
Proceed.
```

Sau đó mới code.

---

# 53. AFTER CODING PROTOCOL

Sau mỗi task, Claude Code phải báo:

```text
Changed:
- file 1
- file 2

Implemented:
- feature 1
- feature 2

Tests:
- test command
- result

Run:
- command

Known limitations:
- ...
```

Không nói "production-ready" nếu chưa kiểm thử.

---

# 54. RESEARCH INTEGRITY

Hệ thống phải phân biệt rõ:

```text
Simulation result
Baseline result
RL result
```

Không được hard-code kết quả đẹp để minh họa.

Không được giả lập metric rồi trình bày như kết quả thực nghiệm.

Không được tuyên bố:

```text
RL tốt hơn baseline
```

trước khi chạy experiment thực tế.

---

# 55. EXPECTED RESEARCH OUTPUT

Cuối V1 phải có khả năng trả lời:

```text
RL có làm giảm waiting time không?

RL có làm giảm total system cost không?

RL có phân phối EV giữa các station tốt hơn nearest-station heuristic không?

RL có làm giảm station overload không?

Hiệu quả có ổn định trên nhiều random seeds không?
```

Các câu hỏi này phải được trả lời bằng metrics thực nghiệm.

---

# 56. MINIMUM VIABLE DEMO

Demo V1 phải thể hiện được:

```text
20 traffic nodes
5 charging stations
50 EVs
```

Trên dashboard:

```text
EVs di chuyển
↓
EV cần sạc
↓
RL chọn charging station
↓
EV di chuyển tới station
↓
Queue thay đổi
↓
EV charging
↓
Metrics cập nhật
```

Người dùng có thể quan sát được hiện tượng congestion.

---

# 57. DEMO COMPARISON

Dashboard nên cho phép lựa chọn:

```text
DQN
Nearest Station
Shortest Travel Time
Least Queue
```

để chạy cùng một scenario.

Mục tiêu là trực quan hóa sự khác biệt giữa các policy.

---

# 58. PERFORMANCE TARGET

V1 phải ưu tiên:

```text
low memory usage
low CPU overhead
fast startup
fast simulation
small model
```

Không đặt mục tiêu benchmark cực đoan.

Mục tiêu thực tế:

> Hệ thống phải chạy mượt trên máy tính cá nhân phổ thông và có thể train DQN với network nhỏ trong thời gian hợp lý.

---

# 59. WHAT AI MUST NOT DO

Không:

```text
- thêm Docker nếu chưa cần
- thêm Redis nếu chưa cần
- thêm Kafka nếu chưa cần
- thêm Celery nếu chưa cần
- thêm Kubernetes
- thêm microservices
- thêm vector database
- thêm LLM
- thêm cloud infrastructure
- thêm Google Maps
- thêm tracking service
- thêm analytics framework
```

trừ khi người dùng yêu cầu hoặc specification được cập nhật.

---

# 60. PRIORITY ORDER

Khi có xung đột giữa các mục tiêu, ưu tiên:

```text
1. Correctness
2. MDP consistency
3. Reproducibility
4. Testability
5. Simplicity
6. Performance
7. UI polish
```

Không hy sinh correctness để có UI đẹp.

Không hy sinh reproducibility để có simulation "trông thật".

---

# 61. INITIAL CLAUDE CODE PROMPT

Khi bắt đầu project, sử dụng prompt:

```text
Đọc toàn bộ PROJECT_SPEC.md.

Không bắt đầu viết toàn bộ project ngay.

Trước tiên hãy:

1. Phân tích architecture.
2. Kiểm tra MDP.
3. Kiểm tra observation space.
4. Kiểm tra action space.
5. Kiểm tra reward.
6. Kiểm tra simulation assumptions.
7. Kiểm tra dependency giữa các module.
8. Xác định mọi ambiguity có thể ảnh hưởng đến implementation.
9. Đề xuất implementation plan theo PHASE 0 → PHASE 10.

Quan trọng:

- Không tự ý thay đổi MDP.
- Không tự ý đổi DQN sang PPO.
- Không tự ý thêm route vào action.
- Không tự ý thêm multi-agent RL.
- Không tự ý thêm công nghệ ngoài stack.
- Không code frontend trước khi simulation contract rõ ràng.

Ở bước này chỉ phân tích và lập kế hoạch.
Chưa triển khai code lớn.
```

---

# 62. PROMPT PHASE 1

Sau khi kế hoạch được duyệt:

```text
Bắt đầu PHASE 1 theo PROJECT_SPEC.md.

Chỉ triển khai traffic network.

Mục tiêu:

- 20 nodes mặc định.
- 5 charging stations.
- node coordinates.
- edge distance.
- traffic weight.
- speed.
- travel time.
- shortest path.

Implement:
backend/simulation/network_graph.py

Đồng thời tạo test phù hợp.

Không triển khai RL.
Không triển khai FastAPI.
Không triển khai frontend.

Sau khi hoàn thành:
- chạy test;
- báo các file thay đổi;
- báo command chạy;
- báo kết quả test;
- không tự chuyển sang PHASE 2.
```

---

# 63. PROMPT PHASE 2

```text
Bắt đầu PHASE 2 theo PROJECT_SPEC.md.

Implement simulation core:

- vehicle.py
- charging_station.py
- traffic_logic.py
- simulator.py

Phải hỗ trợ:

- discrete timestep = 1 simulated second;
- vehicle movement;
- battery consumption;
- traffic;
- station queue;
- charging;
- simulation state.

Không triển khai Gymnasium.
Không triển khai DQN.
Không triển khai FastAPI.
Không triển khai frontend.

Viết test cho các logic quan trọng.

Sau khi hoàn thành chỉ dừng ở PHASE 2.
```

---

# 64. PROMPT PHASE 3

```text
Bắt đầu PHASE 3 theo PROJECT_SPEC.md.

Implement Gymnasium EV environment.

Bắt buộc:

- gymnasium.Env;
- observation_space = Box;
- action_space = Discrete(number_of_stations);
- action chỉ chọn charging station;
- routing sử dụng NetworkX;
- reward đúng specification;
- battery failure penalty;
- station overload không terminate toàn episode;
- reset();
- step();
- termination;
- truncation;
- info.

Sau đó kiểm tra environment bằng test và một policy đơn giản.

Chưa train DQN.

Không triển khai FastAPI hoặc frontend.
```

---

# 65. PROMPT PHASE 4

```text
Bắt đầu PHASE 4.

Implement 3 baseline policies:

1. Nearest Station
2. Shortest Travel Time
3. Least Queue

Các baseline phải dùng cùng simulation interface với RL.

Không thay đổi simulation.

Không thay đổi MDP.

Viết evaluation code cơ bản và test.
```

---

# 66. PROMPT PHASE 5

```text
Bắt đầu PHASE 5.

Implement DQN training bằng Stable-Baselines3.

Requirements:

- MlpPolicy;
- small feed-forward network;
- reproducible random seed;
- configurable training steps;
- model checkpoint;
- save model vào models/;
- không train trong FastAPI.

Trước khi train dài:
1. kiểm tra environment;
2. chạy một training smoke test ngắn;
3. xác nhận model có thể predict action hợp lệ.

Sau đó mới training chính thức.
```

---

# 67. PROMPT PHASE 6

```text
Bắt đầu PHASE 6.

Implement evaluation pipeline.

So sánh:

- DQN
- Nearest Station
- Shortest Travel Time
- Least Queue

Tất cả phải chạy trên cùng scenario và cùng random seeds.

Tính:

- Average Travel Time
- Average Waiting Time
- Total System Cost
- Average Queue Length
- Maximum Queue Length
- Station Utilization
- Failed EVs
- Overload Events
- Episode Reward

Lưu kết quả vào results/.

Không tạo hoặc hard-code kết quả giả.
```

---

# 68. PROMPT PHASE 7

```text
Bắt đầu PHASE 7.

Implement FastAPI backend.

Requirements:

- REST API;
- WebSocket /ws/simulation;
- Pydantic schemas;
- simulation state;
- model inference;
- simulation control.

Training không chạy trong request thread.

Static network data không gửi lặp lại mỗi tick.

Dynamic simulation state gửi qua WebSocket.
```

---

# 69. PROMPT PHASE 8

```text
Bắt đầu PHASE 8.

Implement Next.js frontend.

Requirements:

- App Router;
- TypeScript;
- Tailwind CSS;
- Leaflet / React-Leaflet;
- MapComponent;
- DashboardCharts;
- ControlPanel;
- WebSocket client.

Frontend chỉ visualization và control.

Không đặt RL logic vào frontend.

Không tự tính reward hoặc simulation.
```

---

# 70. FINAL COMMAND

Sau khi toàn bộ project hoàn thành:

```text
Hãy thực hiện FINAL VALIDATION theo PROJECT_SPEC.md.

Không sửa kiến trúc lớn.

Kiểm tra:

1. Backend tests.
2. RL environment.
3. DQN smoke test.
4. DQN evaluation.
5. Baseline evaluation.
6. Reproducibility.
7. FastAPI.
8. WebSocket.
9. Frontend build.
10. End-to-end demo.

Báo cáo:

- PASS
- FAIL
- WARNING

cho từng hạng mục.

Nếu có FAIL, sửa lỗi cần thiết và chạy lại test.

Không tuyên bố project hoàn thành nếu còn lỗi blocking.
```

---

# 71. FINAL PRINCIPLE

Đây là nguyên tắc quan trọng nhất của toàn bộ project:

```text
SIMULATION FIRST
→ ENVIRONMENT SECOND
→ BASELINE THIRD
→ RL FOURTH
→ EVALUATION FIFTH
→ API SIXTH
→ FRONTEND LAST
```

Không đảo ngược thứ tự chỉ để có demo giao diện sớm.

Mục tiêu cuối cùng không phải chỉ là:

```text
"có một dashboard đẹp"
```

mà là:

```text
Có một simulation có kiểm soát
+
Có MDP rõ ràng
+
Có RL agent
+
Có baseline
+
Có evaluation công bằng
+
Có kết quả thực nghiệm
+
Có dashboard trực quan hóa kết quả
```

Đó là nền tảng để dự án vừa có giá trị kỹ thuật, vừa có thể sử dụng làm cơ sở cho báo cáo/đề tài nghiên cứu.