# Cách đọc thí nghiệm: Backbone, CL Method, Protocol, Horizon và Baseline

## Mục đích tài liệu

Trong project này, từ **baseline** xuất hiện ở nhiều ngữ cảnh khác nhau. Nếu không chỉ rõ baseline thuộc trục nào, rất dễ nhầm protocol với model hoặc nhầm data stream với thuật toán continual learning.

Mỗi run phải được hiểu như tổ hợp của năm thành phần độc lập:

\[
\boxed{
\text{Backbone}
\times
\text{CL configuration}
\times
\text{Protocol}
\times
\text{Horizon}
\times
\text{Seed}
}
\]

Ví dụ:

```text
T-DDI
× Replay + Distillation, không merge
× P9 Tail-profile balanced
× H15
× Seed 0
```

Đây mới là một cấu hình thí nghiệm hoàn chỉnh.

---

## 1. Tóm tắt bốn trục chính

| Trục | Các giá trị hiện có | Câu hỏi mà trục này trả lời |
|---|---|---|
| **Backbone/model** | T-DDI, TabM, DDI-GCN | Kiến trúc nào đang học? |
| **CL configuration** | Replay + Distillation; thêm EMA hoặc Selective merge | Model học tuần tự và chống quên bằng cách nào? |
| **Protocol** | P0–P12 | Lớp nào xuất hiện ở task nào? |
| **Horizon** | H8, H15; H29 mới là đề xuất | Chuỗi continual learning gồm bao nhiêu task? |

Seed là trục lặp thí nghiệm để đo độ ổn định, không phải một phương pháp.

Luồng khái niệm:

```text
Dataset DDI2025
      │
      ▼
Backbone/model
      │
      ▼
Continual-learning configuration
      │
      ▼
Data stream do protocol và horizon tạo ra
      │
      ▼
Metrics theo từng task và ở cuối chuỗi
```

Điểm quan trọng nhất:

> P4 và P9 không cạnh tranh với Replay + Distillation. Chúng nằm trên hai trục khác nhau.

---

## 2. Dataset khác backbone

Dataset của project là **DDI2025**, gồm:

- 178 lớp;
- 3.780 đặc trưng đầu vào cho nhánh descriptor;
- train/validation/test được tách không trùng drug pair.

**T-DDI** là một backbone/model được train trên dataset đó. TabM và DDI-GCN là hai backbone khác.

Không nên viết:

```text
Dataset: T-DDI
```

Cách viết đúng:

```text
Dataset: DDI2025
Backbone: T-DDI
```

---

## 3. Learning-method baseline là gì?

Learning-method baseline hiện tại là:

```text
replay_distill_fixed_budget_uniform
```

Tên diễn giải:

> **Replay + Distillation với fixed replay budget và không model merging.**

Contract chính:

- train dữ liệu của task hiện tại;
- replay exemplar của các lớp cũ;
- tổng replay memory: 6.800 exemplar;
- 6.800 replay draw mỗi epoch sau task 0;
- logit distillation từ model của task trước;
- feature distillation bằng MSE;
- checkpoint được chọn bằng validation Macro-F1 trên toàn bộ lớp đã thấy;
- test chỉ dùng để báo cáo.

Luồng học:

```text
Task 0
  └─ train → Model 0

Task 1 + replay từ Task 0 + distillation từ Model 0
  └─ train → Model 1

Task 2 + replay memory + distillation từ Model 1
  └─ train → Model 2

...

Task cuối
  └─ train → Final model
```

Đây là baseline phù hợp khi câu hỏi nghiên cứu là:

> Thêm model merging có cải thiện continual learning so với cách học hiện tại hay không?

Khi đó phép so sánh đúng là:

```text
Replay + Distillation, không merge
                vs
Replay + Distillation + model merging
```

---

## 4. Protocol là gì?

P0, P1, P4, P9 và các ký hiệu tương tự là **data-stream protocol**. Protocol quyết định:

- lớp nào thuộc task nào;
- nhóm head/medium/tail được phân bố ra sao;
- sample mass và long-tail profile biến thiên thế nào qua chuỗi task.

Protocol không trực tiếp thay đổi:

- backbone;
- optimizer;
- replay memory;
- distillation loss;
- số epoch tối đa;
- cách checkpoint được chọn.

Ví dụ giản lược với 12 lớp:

```text
Head:   A=10000, B=8000, C=6000
Medium: D=2000,  E=1500, F=1000
Tail:   G=200,   H=150,  I=100, J=80, K=50, L=20
```

Một protocol có thể tạo stream:

```text
Task 1: A B C D
Task 2: E F G H
Task 3: I J K L
```

Protocol khác có thể tạo:

```text
Task 1: A D G J
Task 2: B E H K
Task 3: C F I L
```

Model và CL method có thể hoàn toàn giống nhau; chỉ data stream thay đổi.

Vì vậy, phát biểu:

> P9 tốt hơn P4.

phải được hiểu là:

> Với cùng backbone, CL method, horizon, split và training contract, stream do P9 tạo ra cho kết quả tốt hơn stream do P4 tạo ra.

Nó không có nghĩa P9 là một thuật toán chống quên tốt hơn Replay + Distillation.

---

## 5. P4 — protocol đối chứng chính

P4 là **Constrained Mass-balanced protocol**.

P4 đồng thời:

- giữ đúng số lớp của từng task;
- phân bổ rarity strata theo quota;
- tối ưu để sample mass giữa các task tương đối cân bằng;
- chỉ dùng thống kê train, không phụ thuộc checkpoint/model.

Ví dụ đơn giản:

```text
A=50000, B=30000, C=20000, D=5000, E=3000, F=1000
```

Một cách chia lệch:

```text
Task 1: A+B+C = 100000
Task 2: D+E+F =   9000
```

P4 hướng tới cách chia cân bằng hơn:

```text
Task 1: A+E+F ≈ 54000
Task 2: B+C+D ≈ 55000
```

P4 được chọn làm **protocol đối chứng chính** vì có stream tương đối cân bằng, dễ tái lập và không dùng tín hiệu từ model để xây lịch.

Trong T-DDI H8, đủ năm seed:

| Metric | P4 |
|---|---:|
| Macro-F1 | 0.3825 ± 0.0246 |
| Balanced Accuracy | 0.4795 ± 0.0250 |
| Task Forgetting | 0.3562 ± 0.0283 |

---

## 6. P9 — Tail-profile balanced

P9 cũng chia 178 lớp thành các task nhưng kiểm soát thêm **hình dạng long-tail** của mỗi task.

Trong implementation hiện tại, các lớp được chia thành bốn quantile theo thứ hạng `log(train_count)`:

```text
Q1 → phía head
Q2
Q3
Q4 → phía tail
```

P9 cố gắng:

- giữ profile Q1–Q4 tương đối nhất quán giữa các task;
- cân bằng sample mass;
- giữ đúng số lớp/task;
- không dùng test để xây schedule.

Hai task có thể có cùng sample mass nhưng profile rất khác:

```text
Task A: 9500 + 300 + 200 = 10000
Task B: 3500 + 3300 + 3200 = 10000
```

Chỉ cân bằng tổng mass chưa phản ánh được việc Task A bị một lớp thống trị. P9 được thiết kế để kiểm soát thêm khác biệt này.

So sánh H8, năm seed:

| Metric | P4 baseline | P9 | P9 − P4 |
|---|---:|---:|---:|
| Macro-F1 ↑ | 0.3825 | **0.3877** | +0.0052 |
| Balanced Accuracy ↑ | 0.4795 | **0.4870** | +0.0074 |
| Forgetting ↓ | 0.3562 | **0.3489** | −0.0073 |
| Tail F1 ↑ | 0.4801 | **0.4928** | +0.0127 |

P9 có xu hướng nhỉnh hơn P4, nhưng khác biệt chưa có ý nghĩa thống kê với năm seed.

---

## 7. Tại sao cả P1 và P4 đều từng được gọi là baseline?

Baseline chỉ có nghĩa khi gắn với một câu hỏi nghiên cứu cụ thể.

| Câu hỏi | Baseline cần dùng |
|---|---|
| Một protocol cân bằng đơn giản hoạt động thế nào? | **P1 Frequency-balanced** |
| P9 có tốt hơn protocol đối chứng trung lập không? | **P4 Constrained Mass-balanced** |
| Model merging có giúp chống quên không? | **Replay + Distillation, không merge** |
| Backbone mới có tốt hơn T-DDI không? | **T-DDI dưới cùng protocol và CL contract** |

Do đó không nên viết đơn độc:

```text
baseline = P4
```

Nên viết đầy đủ:

```text
Protocol baseline for the P9 comparison: P4
```

hoặc:

```text
No-merging CL baseline: Replay + Distillation
```

---

## 8. Horizon là gì?

Horizon cho biết 178 lớp được trải qua bao nhiêu task liên tiếp.

### H8

```text
[38, 20, 20, 20, 20, 20, 20, 20]
```

- 8 task;
- task đầu có 38 lớp;
- 7 task sau, mỗi task có 20 lớp.

### H15

```text
[38, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
```

- 15 task;
- task đầu vẫn có 38 lớp;
- 14 task sau, mỗi task có 10 lớp.

### H29 — nhánh mở rộng đã chạy

```text
[38, 5 × 28]
```

- 29 task;
- đã có run P9-H29 seed 0–2 cho C3 và C4; classifier correction C4+GroupPrior được đánh giá trên cùng checkpoint C4. Xem `docs/C4_GROUPPRIOR_PROGRESS.md`. Đây là nhánh xác nhận 3 seed, chưa phải bảng 5 seed.

Tổng số lớp luôn là 178. Horizon dài hơn tạo nhiều lần chuyển task, cập nhật replay buffer, distillation và consolidation hơn.

P9-H8 và P9-H15 vì vậy là hai experiment khác nhau:

```text
P9-H8  = P9 protocol + 8-task horizon
P9-H15 = P9 protocol + 15-task horizon
```

### Cảnh báo khi so H8 với H15

Các run H15 hiện giữ nguyên replay draws theo mỗi epoch/task. Vì H15 có nhiều task transition và replay phase hơn H8, phép so H8–H15 **chưa được kiểm soát theo tổng compute**.

Do đó:

- có thể dùng H15 để quan sát hành vi khi chuỗi dài hơn;
- chưa nên quy toàn bộ chênh lệch H8–H15 chỉ cho horizon;
- một study horizon chính thức cần khóa tổng optimizer steps hoặc tổng replay exposure, hoặc báo cáo riêng cả hai đại lượng.

---

## 9. Model merging nằm ở đâu?

Model merging là phần mở rộng của CL configuration, không phải protocol.

### 9.1 No-merge baseline

```text
Replay + Distillation
→ dùng trực tiếp checkpoint tốt nhất của task hiện tại
```

### 9.2 EMA-backbone merge

Sau khi train task hiện tại:

\[
\theta_{merged} = (1-\alpha)\theta_{old} + \alpha\theta_{trained}
\]

Trong pilot:

- chỉ merge các tham số backbone dùng chung;
- classifier lấy từ model vừa train;
- `alpha=0.5` nghĩa là backbone sau merge nhận 50% trọng số cũ và 50% trọng số mới.

### 9.3 Selective merge

Selective merge:

- merge backbone;
- merge classifier rows của lớp cũ;
- giữ classifier rows của lớp mới từ model vừa train.

Mục tiêu là giữ kiến thức cũ mà không làm hỏng phần classifier mới. Pilot cho thấy `alpha=0.5` vẫn quá mạnh và làm suy giảm plasticity nghiêm trọng.

---

## 10. Vì sao pilot H15 có sáu run?

Pilot kết hợp hai protocol với ba CL configuration:

| | No merge | EMA backbone | Selective merge |
|---|:---:|:---:|:---:|
| **P4** | ✓ | ✓ | ✓ |
| **P9** | ✓ | ✓ | ✓ |

Số run:

```text
2 protocol × 3 CL configuration × 1 seed = 6 run
```

Danh sách đầy đủ:

| Protocol | CL configuration | Horizon | Seed |
|---|---|---|---:|
| P4 | Replay + Distillation, không merge | H15 | 0 |
| P4 | Replay + Distillation + EMA backbone | H15 | 0 |
| P4 | Replay + Distillation + Selective merge | H15 | 0 |
| P9 | Replay + Distillation, không merge | H15 | 0 |
| P9 | Replay + Distillation + EMA backbone | H15 | 0 |
| P9 | Replay + Distillation + Selective merge | H15 | 0 |

---

## 11. Kết quả pilot H15 hiện tại

Tất cả số dưới đây là **seed 0**, chưa phải kết quả đa seed.

| Protocol | CL configuration | Macro-F1 ↑ | Balanced Acc. ↑ | Forgetting ↓ | Head F1 ↑ | Tail F1 ↑ |
|---|---|---:|---:|---:|---:|---:|
| P4 | **No-merge baseline** | **0.3474** | 0.4678 | 0.3137 | 0.1115 | **0.4466** |
| P4 | EMA backbone, α=0.5 | 0.3062 | **0.4893** | **0.2040** | **0.1372** | 0.3412 |
| P4 | Selective, α=0.5 | 0.2201 | 0.2282 | 0.4183 | 0.0442 | 0.3374 |
| P9 | **No-merge baseline** | **0.3616** | 0.4734 | 0.2880 | 0.1214 | **0.4602** |
| P9 | EMA backbone, α=0.5 | 0.3025 | **0.5199** | **0.1363** | **0.1530** | 0.3188 |
| P9 | Selective, α=0.5 | 0.1626 | 0.1954 | 0.4851 | 0.0428 | 0.2321 |

### 11.1 P9-H15 baseline so với P4-H15 baseline

| Metric | P4 | P9 | P9 − P4 |
|---|---:|---:|---:|
| Macro-F1 | 0.3474 | **0.3616** | +0.0142 |
| Balanced Accuracy | 0.4678 | **0.4734** | +0.0056 |
| Forgetting | 0.3137 | **0.2880** | −0.0257 |
| Head F1 | 0.1115 | **0.1214** | +0.0099 |
| Tail F1 | 0.4466 | **0.4602** | +0.0136 |

Trong seed 0, P9 baseline nhỉnh hơn P4 trên toàn bộ các metric ở bảng này. Cần thêm seed trước khi kết luận thống kê.

### 11.2 EMA thể hiện stability–plasticity trade-off

Trên P9:

```text
Macro-F1:   0.3616 → 0.3025  giảm
Forgetting: 0.2880 → 0.1363  cải thiện mạnh
Head F1:    0.1214 → 0.1530  tăng
Tail F1:    0.4602 → 0.3188  giảm mạnh
```

Diễn giải:

```text
giữ model cũ mạnh hơn
       ├─ stability tăng: forgetting giảm, Head F1 tăng
       └─ plasticity giảm: Macro-F1 và Tail F1 giảm
```

EMA không thể được tuyên bố là tốt hơn chỉ vì forgetting thấp hơn; nó đánh đổi đáng kể khả năng học/adapt các lớp mới.

### 11.3 Selective merge α=0.5 thất bại

Selective merge làm giảm mạnh:

- Macro-F1;
- Balanced Accuracy;
- Head F1;
- Tail F1;

đồng thời forgetting còn tăng. Cấu hình này không nên được mở rộng sang nhiều seed.

Validation Macro-F1 sau merge giảm ở 14/14 task của cả P4 và P9. Đây là bằng chứng trực tiếp rằng `alpha=0.5` kéo model về trạng thái cũ quá mạnh.

---

## 12. Cách diễn giải một run cụ thể

### P9-H15 baseline seed 0

```text
Dataset:          DDI2025, 178 lớp
Backbone:         T-DDI
CL method:        Replay + Distillation
Model merging:    Không
Protocol:         P9 Tail-profile balanced
Horizon:          15 task
Training seed:    0
```

Kết quả:

```text
Macro-F1       = 0.3616
Balanced Acc.  = 0.4734
Forgetting     = 0.2880
Head F1        = 0.1214
Tail F1        = 0.4602
```

### P9-H15 EMA seed 0

Mọi thành phần giữ nguyên, chỉ thay:

```text
Model merging: Không
```

bằng:

```text
Model merging: EMA backbone, alpha=0.5
```

Như vậy chênh lệch giữa hai run có thể được quy cho EMA merging trong phạm vi seed 0 và contract H15 này.

---

## 13. Quy ước đặt tên và báo cáo

### 13.1 Tên cấu hình đầy đủ

Nên dùng mẫu:

```text
<backbone>-<protocol>-<horizon>-<cl_variant>-seed<seed>
```

Ví dụ:

```text
tddi-p9-h15-no_merge-seed0
tddi-p9-h15-ema_backbone-alpha0p5-seed0
tddi-p4-h15-selective-alpha0p5-seed0
```

### 13.2 Baseline trong bảng kết quả

Mỗi bảng phải có dòng mô tả:

```text
Baseline axis: CL configuration
Baseline: Replay + Distillation, no post-task merging
```

hoặc:

```text
Baseline axis: Protocol
Baseline: P4 Constrained Mass-balanced
```

### 13.3 Phát biểu nên dùng

Đúng:

> Với T-DDI, Replay + Distillation và H15 seed 0, P9 đạt Macro-F1 cao hơn P4 0.0142.

Đúng:

> Với P9-H15 seed 0, EMA α=0.5 giảm forgetting nhưng làm giảm Macro-F1 và Tail F1.

Không đủ rõ:

> P9 tốt hơn baseline.

Không đúng loại so sánh:

> P9 tốt hơn Replay + Distillation.

---

## 14. Hướng thí nghiệm tiếp theo

Không tiếp tục Selective merge `alpha=0.5` sang seed 1–4.

Nếu tiếp tục model merging, nhánh hợp lý nhất là EMA backbone với:

```text
alpha ∈ {0.75, 0.90, 1.00}
```

Trong định nghĩa hiện tại:

- alpha càng gần 1 càng ưu tiên model vừa train;
- `alpha=1.0` tương đương không merge;
- candidate alpha phải được chọn bằng validation, không dùng test;
- nếu không candidate nào tốt hơn `alpha=1.0`, giữ nguyên no-merge model.

Trước khi mở rộng năm seed, nên yêu cầu:

- validation Macro-F1 sau merge không giảm có hệ thống;
- final Macro-F1 không giảm đáng kể;
- Tail F1 không bị hy sinh quá lớn;
- forgetting hoặc BWT cải thiện;
- báo cáo cả stability và plasticity thay vì chỉ chọn metric có lợi cho merging.

---

## 15. Bốn câu để nhớ

> **Backbone/model:** cái gì đang học?

> **CL method/configuration:** nó học tuần tự và chống quên bằng cách nào?

> **Protocol P4/P9:** nó nhìn thấy các lớp theo cấu trúc nào?

> **Horizon H8/H15/H29:** nó phải học liên tục qua bao nhiêu chặng?

Tóm lại:

> P4/P9 không cạnh tranh với baseline Replay + Distillation. P4 chỉ là baseline khi trục đang nghiên cứu là protocol. Khi trục đang nghiên cứu là model merging, baseline phải là Replay + Distillation không merge.

---

## 16. Artifact liên quan

- Contract protocol T-DDI: `configs/protocol_study_tddi.json`
- Contract long-tail P9–P12: `configs/long_tail_protocol_study.json`
- Contract pilot H15 merging: `configs/h15_model_merging_pilot.json`
- Kết quả H8 P0–P8: `docs/RESULTS.md`
- Kết quả backbone: `docs/BACKBONE_STUDY.md`
- Kết quả P9 năm seed: `outputs/runs_long_tail/analysis_p9_all_seeds/`
- Kết quả H15 merging: `outputs/runs_h15_merging/`
