# C4+GroupPrior: tiến độ xác nhận

Tính đến 27/09/2026, `C4-GroupPrior-v1` là nhánh hiệu chỉnh classifier sau khi train C4 X-DER thích nghi. Candidate và quy tắc đã khóa ở `configs/c4_grouprior_candidate.json`: chia lớp theo số mẫu train thành Head (`>1000`), Medium (`101–1000`) và Tail (`<=100`); sau mỗi task ước lượng tỷ lệ dự đoán so với tỷ lệ thật của từng nhóm trên validation, rồi cộng bias nhóm vào logit với `lambda=1.0`. Test chỉ dùng để đánh giá.

## Kết quả đã có

| Phạm vi | Seed | Kết quả so với C3 |
|---|---:|---|
| P4-H8, P4-H15, P9-H15 | 0–2 mỗi setting | Macro-F1 test tăng ở cả 9/9 cặp; mức tăng trung bình lần lượt `+0.0355`, `+0.0579`, `+0.0445`. |
| P9-H29 | 0–4 | Macro-F1 `0.3874 ± 0.0153`; tăng `+0.0697 ± 0.0152` so với C3 và thắng 5/5 cặp. |
| P4-H29 | 0–4 | Macro-F1 `0.3897 ± 0.0139`; tăng `+0.0602 ± 0.0198` so với C3 và thắng 5/5 cặp. |
| D1-H29 | 0–4 | Macro-F1 `0.3990 ± 0.0201`; tăng `+0.0768 ± 0.0368` so với C3 và thắng 5/5 cặp. |

Nguồn: `outputs/runs_c4_grouprior_screening/analysis/three_seed_gain_summary.csv`, `outputs/runs_c4_grouprior_h29/analysis/h29_five_seed_aggregate.csv` và `outputs/runs_c4_grouprior_p4_h29/analysis/h29_five_seed_aggregate.csv`. Run thô nằm trong các thư mục tương ứng dưới `outputs/`; phần lớn không được đưa lên Git vì dung lượng lớn.

### P4-H29, năm seed

| Method | Macro-F1 ↑ | Balanced acc. ↑ | Head F1 ↑ | Medium F1 ↑ | Tail F1 ↑ | Forgetting ↓ |
|---|---:|---:|---:|---:|---:|---:|
| C3 | 0.3295 ± 0.0159 | 0.4398 ± 0.0207 | 0.1102 ± 0.0139 | 0.3406 ± 0.0132 | 0.4241 ± 0.0399 | 0.2925 ± 0.0259 |
| C4 thô | 0.3034 ± 0.0090 | **0.5852 ± 0.0222** | 0.1874 ± 0.0157 | 0.3768 ± 0.0167 | 0.3092 ± 0.0158 | **0.2668 ± 0.0181** |
| C4+GroupPrior | **0.3897 ± 0.0139** | 0.4641 ± 0.0197 | **0.1954 ± 0.0131** | **0.3861 ± 0.0094** | **0.4823 ± 0.0208** | 0.2945 ± 0.0232 |

### P9-H29, năm seed

| Method | Macro-F1 ↑ | Balanced acc. ↑ | Head F1 ↑ | Medium F1 ↑ | Tail F1 ↑ | Forgetting ↓ |
|---|---:|---:|---:|---:|---:|---:|
| C3 | 0.3178 ± 0.0146 | 0.4533 ± 0.0350 | 0.1063 ± 0.0139 | 0.3479 ± 0.0165 | 0.3963 ± 0.0342 | 0.2778 ± 0.0479 |
| C4 thô | 0.2868 ± 0.0264 | **0.5932 ± 0.0534** | 0.1862 ± 0.0245 | 0.3635 ± 0.0143 | 0.2834 ± 0.0581 | **0.2542 ± 0.0455** |
| C4+GroupPrior | **0.3874 ± 0.0153** | 0.4711 ± 0.0150 | **0.1936 ± 0.0278** | **0.3934 ± 0.0092** | **0.4736 ± 0.0212** | 0.2551 ± 0.0204 |

### Pilot chuyển backbone, P4-H8 seed 0

| Backbone | C3 Macro-F1 | C4 thô Macro-F1 | C4+GroupPrior Macro-F1 | Kết luận pilot |
|---|---:|---:|---:|---|
| TabM | **0.3792** | 0.2090 | 0.1552 | GroupPrior giảm thêm so với C4 và thấp hơn C3. |
| DDI-GCN | **0.3669** | 0.2616 | 0.2692 | GroupPrior tăng nhẹ so với C4 nhưng thấp hơn C3. |

Jobs `22161` và `22163` hoàn tất trên `mantis-05` (RTX A6000). Artifact pilot nằm tại `outputs/runs_c4_grouprior_backbones/analysis/`. Kết quả không đạt điều kiện mở rộng sang nhiều seed vì candidate không thắng C3 trên cả hai backbone.

### Xác nhận lịch động D1-H29

| Method | Macro-F1 ↑ | Balanced acc. ↑ | Head F1 ↑ | Medium F1 ↑ | Tail F1 ↑ | Forgetting ↓ | BWT ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| C3 | 0.3223 ± 0.0326 | 0.4773 ± 0.0546 | 0.1169 ± 0.0238 | 0.3443 ± 0.0215 | 0.4031 ± 0.0693 | 0.3056 ± 0.0542 | **−0.1926 ± 0.0458** |
| C4 thô | 0.2908 ± 0.0422 | **0.5995 ± 0.0668** | 0.1972 ± 0.0309 | 0.3670 ± 0.0411 | 0.2844 ± 0.0763 | **0.2749 ± 0.0456** | −0.2753 ± 0.0510 |
| C4+GroupPrior | **0.3990 ± 0.0201** | 0.4848 ± 0.0247 | **0.2033 ± 0.0361** | **0.4019 ± 0.0140** | **0.4880 ± 0.0294** | 0.2888 ± 0.0329 | −0.2790 ± 0.0339 |

Jobs `22182` và `22183` đã hoàn tất trên `mantis-05`. GroupPrior tăng Macro-F1 `+0.0768 ± 0.0368` và new-task F1 `+0.1167 ± 0.0362` so với C3, thắng 5/5 seed ở cả hai metric. Balanced accuracy cao hơn C3 ở 3/5 seed. Forgetting thấp hơn ở 3/5 seed và giảm trung bình `0.0168`; BWT thấp hơn C3 ở 5/5 seed. Aggregate nằm tại `outputs/runs_c4_grouprior_dynamic/d1/analysis/d1_five_seed_aggregate.csv`.

## Diagnostic cho candidate tiếp theo

Bốn phép thử dùng D1-H29 seed 0 và backbone T-DDI để tách giới hạn representation, replay và classifier:

| Diagnostic | Trạng thái | Chi tiết |
|---|---|---|
| Joint offline T-DDI | Hoàn tất | Macro-F1 `0.8179`, balanced accuracy `0.8195`; nguồn `outputs/runs_c3_c4_diagnostics/j0_joint_natural_seed0`. |
| Unlimited replay H29 | Job `22368`, hoàn tất | Macro-F1 **0.6490**, BA **0.8946**; giữ 520,841 mẫu của đủ 178 lớp. |
| C4 + frozen balanced classifier | Job `22367`, hoàn tất | Macro-F1 `0.5525`; đóng băng C4 và train linear head bằng inverse-frequency sampling. |
| C4 + dynamic class-level prior | Job `22367`, hoàn tất | Macro-F1 `0.4145`; hệ số chọn trên validation là `0.75`. |

Kết hợp frozen balanced classifier với GroupPrior đạt Macro-F1 **0.5833**. Trên cùng seed, C4 thô là `0.2585` và C4+GroupPrior là `0.4239`. Unlimited replay + distillation cao hơn frozen head + GroupPrior `+0.0657`, cho thấy nút thắt chính nằm ở replay/retention trong quá trình học liên tục; classifier prior vẫn có lợi nhưng chưa thay thế được việc giữ dữ liệu lịch sử.

Output chung nằm dưới `outputs/runs_c4_next_candidate_diagnostics/`. Contract máy đọc được nằm tại `configs/c4_next_candidate_diagnostics.json`. Kết quả này sẽ quyết định có ghép dynamic prior, balanced classifier và adaptive replay thành candidate mới hay không.

## Giới hạn diễn giải

- P4-H29 và P9-H29 đã đủ 5 seed. P4-H15 và P9-H15 vẫn mới có 3 seed.
- Trên P4-H29, GroupPrior thắng về Macro-F1 nhưng C4 thô vẫn tốt nhất về balanced accuracy và forgetting.
- Trên P9-H29, C4+GroupPrior tăng Macro-F1 `+0.0697 ± 0.0152` và thắng 5/5 seed; balanced accuracy tăng trung bình `+0.0178` và forgetting giảm trung bình `0.0228` so với C3.
- Pilot TabM/DDI-GCN chỉ có một seed mỗi backbone. Nó đủ để bác bỏ việc mở rộng tự động candidate hiện tại, nhưng chưa đủ để kết luận chung rằng mọi dạng hiệu chỉnh prior đều không chuyển backbone.
- D1-H29 xác nhận lợi ích Macro-F1 qua 5/5 thứ tự lớp ngẫu nhiên, nhưng BWT giảm ở 5/5 seed; candidate ưu tiên hiệu năng endpoint và plasticity hơn backward transfer.
- Đánh giá hiệu chỉnh theo từng task trên ba setting đầu cho thấy retention/forgetting tốt hơn C3, nhưng new-task F1 thấp hơn cả C3 và C4 thô. Xem `outputs/runs_c4_grouprior_screening/per_task_analysis/retention_three_seed_summary.csv`.
- H8, H15 và H29 là các horizon khác nhau; chưa kiểm soát tổng compute khi so trực tiếp giữa chúng.

Các script tái lập gồm `scripts/run_c4_classifier_diagnostics.py`, `scripts/screen_c4_grouprior_robustness.py`, `scripts/run_c4_grouprior_seed12_confirmation.sh`, `scripts/run_c4_grouprior_h29_seed0.sh`, `scripts/run_c4_grouprior_h29_seed12.sh`, `scripts/aggregate_c4_grouprior_h29.py`, `scripts/evaluate_c4_grouprior_p4_h8_backbone.py` và hai launcher Slurm nêu trên.
