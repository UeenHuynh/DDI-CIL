# C4+GroupPrior: tiến độ xác nhận

Tính đến 16/09/2026, `C4-GroupPrior-v1` là nhánh hiệu chỉnh classifier sau khi train C4 X-DER thích nghi. Candidate và quy tắc đã khóa ở `configs/c4_grouprior_candidate.json`: chia lớp theo số mẫu train thành Head (`>1000`), Medium (`101–1000`) và Tail (`<=100`); sau mỗi task ước lượng tỷ lệ dự đoán so với tỷ lệ thật của từng nhóm trên validation, rồi cộng bias nhóm vào logit với `lambda=1.0`. Test chỉ dùng để đánh giá.

## Kết quả đã có

| Phạm vi | Seed | Kết quả so với C3 |
|---|---:|---|
| P4-H8, P4-H15, P9-H15 | 0–2 mỗi setting | Macro-F1 test tăng ở cả 9/9 cặp; mức tăng trung bình lần lượt `+0.0355`, `+0.0579`, `+0.0445`. |
| P9-H29 | 0–2 | Macro-F1 test tăng trung bình `+0.0688 ± 0.0069` (sample standard deviation), thắng 3/3 cặp. |

Nguồn: `outputs/runs_c4_grouprior_screening/analysis/three_seed_gain_summary.csv` và `outputs/runs_c4_grouprior_h29/analysis/h29_three_seed_paired_delta_summary.csv`. Run thô nằm trong `outputs/runs_c4_grouprior_screening/` và `outputs/runs_c4_grouprior_h29/`; các thư mục run không được đưa lên Git vì dung lượng lớn.

## Giới hạn diễn giải

- Nhánh xác nhận hiện có 3 seed mỗi setting; bảng kết quả cuối cùng 5 seed còn chờ.
- Trên P9-H29, C4+GroupPrior tăng Macro-F1 nhưng task forgetting trung bình chỉ giảm `0.0316` so với C3 và balanced accuracy tăng `0.0256`; cần báo cáo các metric song song thay vì chỉ Macro-F1.
- Đánh giá hiệu chỉnh theo từng task trên ba setting đầu cho thấy retention/forgetting tốt hơn C3, nhưng new-task F1 thấp hơn cả C3 và C4 thô. Xem `outputs/runs_c4_grouprior_screening/per_task_analysis/retention_three_seed_summary.csv`.
- H8, H15 và H29 là các horizon khác nhau; chưa kiểm soát tổng compute khi so trực tiếp giữa chúng.

Các script tái lập gồm `scripts/run_c4_classifier_diagnostics.py`, `scripts/screen_c4_grouprior_robustness.py`, `scripts/run_c4_grouprior_seed12_confirmation.sh`, `scripts/run_c4_grouprior_h29_seed0.sh`, `scripts/run_c4_grouprior_h29_seed12.sh` và `scripts/aggregate_c4_grouprior_h29.py`.
