# DDI2025-CIL — tổng hợp đầy đủ kết quả

**Cập nhật: 16/09/2026.** Tài liệu này gom các kết quả hiện có trong workspace. `P` là protocol chia lớp, `H` là số task (horizon), `C` là phương pháp học liên tục. Tất cả metric ở **task cuối trên tập test** trừ khi ghi khác. `↑` là cao hơn tốt hơn; `↓` là thấp hơn tốt hơn. Dấu `±` biểu diễn **sample standard deviation** giữa các seed, không phải khoảng tin cậy. Task forgetting lấy dòng `mean_old_tasks` của `forgetting.csv` hoặc bảng tổng hợp tương ứng. Không so trực tiếp giữa các bảng có protocol, horizon, backbone, track huấn luyện hoặc ngân sách khác nhau.

## 1. Bản đồ thí nghiệm và trạng thái

| Nhánh | Cấu hình | Trạng thái | Nguồn chính |
|---|---|---|---|
| Protocol P0–P8 | H8, T-DDI, replay + distillation cố định | 9 protocol × 5 seed | `docs/RESULTS.md`, `outputs/runs_backbones/` |
| Backbone | P0/P2/P3/P4/P6-H8, TabM và DDI-GCN | 50/50 run, 5 seed mỗi ô | `README.md`, `outputs/runs_selected_backbones/` |
| Long-tail P9 | H8, T-DDI/TabM/DDI-GCN | 5 seed mỗi backbone | `outputs/runs_long_tail/analysis_p9_all_seeds/`, `outputs/runs_p9_backbones/` |
| Long-tail P10–P12 | H8, T-DDI | Pilot seed 0 | `outputs/runs_long_tail/` |
| Phương pháp C0–C6 | P9-H8, T-DDI | 5 seed cho mỗi method đã chạy; C7 chưa chạy | `outputs/runs_p9_cl_methods/analysis/method_summary.csv` |
| Model merging | P4/P9-H15, T-DDI | Pilot seed 0, 6 run | `outputs/runs_h15_merging/` |
| C4+GroupPrior | P4-H8, P4-H15, P9-H15, P9-H29 | Xác nhận 3 seed mỗi setting | `outputs/runs_c4_grouprior_screening/`, `outputs/runs_c4_grouprior_h29/` |
| Lịch động D1-H29 | T-DDI, C3/C4/C4+GroupPrior | Hoàn tất seed 0 | `outputs/runs_c4_grouprior_dynamic/d1/` |

H8 dùng `[38, 20 × 7]`; H15 dùng `[38, 10 × 14]`; H29 dùng `[38, 5 × 28]`. Mỗi lịch có 178 lớp. Ngân sách replay cấp theo task khiến tổng compute khác nhau giữa các horizon; vì vậy chưa thể quy chênh lệch H8/H15/H29 chỉ cho độ dài chuỗi.

## 2. Protocol gốc P0–P8: T-DDI, H8, 5 seed

Chỉ lịch đến của lớp thay đổi. Backbone, method, split, task layout, optimizer và replay budget được giữ cố định. Các số là mean ± standard deviation trên seed 0–4.

| Protocol | Macro-F1 ↑ | Balanced acc. ↑ | Accuracy ↑ | Weighted-F1 ↑ | Forgetting ↓ |
|---|---:|---:|---:|---:|---:|
| P0 Random | 0.3769 ± 0.0194 | 0.5030 ± 0.0390 | 0.2196 ± 0.0621 | 0.1511 ± 0.0432 | 0.3213 ± 0.0418 |
| P1 Frequency-balanced | 0.3585 ± 0.0081 | 0.5189 ± 0.0153 | 0.1215 ± 0.0037 | 0.0754 ± 0.0036 | 0.3236 ± 0.0126 |
| P2 Head→tail | 0.2719 ± 0.0050 | **0.7642 ± 0.0170** | 0.4645 ± 0.0119 | 0.5452 ± 0.0120 | **0.0441 ± 0.0011** |
| P3 Tail→head | **0.4935 ± 0.0090** | 0.4530 ± 0.0132 | **0.7996 ± 0.0013** | **0.7622 ± 0.0021** | 0.5003 ± 0.0107 |
| P4 Mass-balanced | 0.3825 ± 0.0246 | 0.4795 ± 0.0250 | 0.1621 ± 0.0140 | 0.0929 ± 0.0113 | 0.3562 ± 0.0283 |
| P5 Multi-factor | 0.3791 ± 0.0167 | 0.4785 ± 0.0170 | 0.1601 ± 0.0192 | 0.0888 ± 0.0121 | 0.3570 ± 0.0270 |
| P6 Difficulty-balanced | 0.3831 ± 0.0184 | 0.4947 ± 0.0191 | 0.1662 ± 0.0156 | 0.0963 ± 0.0130 | 0.3438 ± 0.0324 |
| P7 Confusion-spread | 0.3805 ± 0.0206 | 0.4792 ± 0.0140 | 0.1651 ± 0.0149 | 0.0934 ± 0.0111 | 0.3668 ± 0.0143 |
| P8 Rarity-drift | 0.3759 ± 0.0238 | 0.4788 ± 0.0136 | 0.1814 ± 0.0262 | 0.1094 ± 0.0164 | 0.3529 ± 0.0127 |

P2/P3 là stress test của thứ tự lớp: P2 có balanced accuracy và forgetting tốt nhất nhưng Macro-F1 thấp; P3 có Macro-F1 cao nhất nhưng forgetting cao nhất. P4 được chọn làm protocol chính vì chia task không phụ thuộc checkpoint/model; P6 tốt nhất trong nhóm P4–P8 theo Macro-F1, balanced accuracy và forgetting nhưng dùng tín hiệu từ T-DDI. Nguồn số đầy đủ: `docs/RESULTS.md`; cấu hình: `configs/protocol_study_tddi.json`.

## 3. Backbone study: TabM và DDI-GCN, H8, 5 seed

Method chung là replay + distillation; mỗi ô gồm seed 0–4. P6 kế thừa lịch model-informed tạo từ T-DDI, nên nhánh P6 không phải phép so backbone trung lập.

| Backbone | Protocol | Macro-F1 ↑ | Balanced acc. ↑ | Forgetting ↓ |
|---|---|---:|---:|---:|
| TabM | P4 | 0.3659 ± 0.0123 | 0.4777 ± 0.0172 | 0.3311 ± 0.0138 |
| TabM | P6 | 0.3665 ± 0.0195 | 0.4839 ± 0.0228 | 0.3404 ± 0.0390 |
| TabM | P0 | 0.3539 ± 0.0244 | 0.5152 ± 0.0558 | 0.3071 ± 0.0538 |
| TabM | P2 | 0.1801 ± 0.0029 | 0.6897 ± 0.0072 | 0.0821 ± 0.0019 |
| TabM | P3 | 0.4164 ± 0.0036 | 0.3908 ± 0.0025 | 0.5794 ± 0.0068 |
| DDI-GCN | P4 | 0.4116 ± 0.0339 | 0.5308 ± 0.0362 | 0.1155 ± 0.0267 |
| DDI-GCN | P6 | 0.4256 ± 0.0425 | 0.5390 ± 0.0577 | 0.1315 ± 0.0367 |
| DDI-GCN | P0 | 0.4167 ± 0.0268 | 0.5555 ± 0.0604 | 0.1327 ± 0.0869 |
| DDI-GCN | P2 | 0.2760 ± 0.0065 | 0.7786 ± 0.0384 | 0.0538 ± 0.0041 |
| DDI-GCN | P3 | 0.5965 ± 0.0112 | 0.5503 ± 0.0122 | 0.3998 ± 0.0146 |

P3 có Macro-F1 cao nhất và P2 có balanced accuracy cao nhất, forgetting thấp nhất trên cả hai backbone. Nguồn: `README.md` (bảng study backbone), `outputs/runs_selected_backbones/`; phương pháp: `docs/BACKBONE_STUDY.md`.

## 4. Long-tail P9–P12

P9 là Tail-Profile Balanced. P10 là Effective-Number Balanced (`beta=0.999` trong pilot), P11 là Progressive Tail-Drift, P12 là Tail-Shock ở task 3. Study này riêng với P0–P8.

| Protocol / backbone | Seed | Macro-F1 ↑ | Balanced acc. ↑ | Forgetting ↓ | Mức bằng chứng |
|---|---:|---:|---:|---:|---|
| P9 / T-DDI | 0–4 | 0.3877 ± 0.0126 | 0.4870 ± 0.0201 | 0.3489 ± 0.0262 | 5 seed |
| P9 / TabM | 0–4 | 0.3625 ± 0.0135 | 0.4681 ± 0.0123 | 0.3409 ± 0.0135 | 5 seed |
| P9 / DDI-GCN | 0–4 | 0.4103 ± 0.0283 | 0.4977 ± 0.0660 | 0.1622 ± 0.0541 | 5 seed |
| P10 / T-DDI | 0 | 0.3884 | 0.5110 | 0.3525 | Pilot 1 seed |
| P11 / T-DDI | 0 | 0.3608 | 0.4697 | 0.3648 | Pilot 1 seed |
| P12 / T-DDI | 0 | 0.3920 | 0.5102 | 0.3316 | Pilot 1 seed |

Các ô P9/TabM và P9/DDI-GCN được tính từ `metrics.csv` dòng `test_seen_all` cuối và `forgetting.csv` dòng `mean_old_tasks` của năm run trong `outputs/runs_p9_backbones/`; P10–P12 từ run seed 0 tương ứng trong `outputs/runs_long_tail/`. P9/T-DDI từ `outputs/runs_long_tail/analysis_p9_all_seeds/long_tail_aggregate_summary.csv`. Pilot P10–P12 chỉ cho biết pipeline chạy được và hướng quan sát ban đầu; chưa đủ để kết luận thứ hạng. P9/T-DDI có Head F1 `0.1539 ± 0.0150`, Medium F1 `0.3929 ± 0.0080`, Tail F1 `0.4928 ± 0.0247`.

## 5. So phương pháp C trên P9-H8, T-DDI, 5 seed

Track offline và track một lượt qua stream có ngân sách khác nhau. C0O/C1O là đối chứng một lượt; C5 là OTC/MMOT thích nghi. C7 DEMD chưa có run. Bảng dưới dùng `outputs/runs_p9_cl_methods/analysis/method_summary.csv`.

| Track | Method | Mô tả | Macro-F1 ↑ | Balanced acc. ↑ | Forgetting ↓ | Head F1 | Tail F1 |
|---|---|---|---:|---:|---:|---:|---:|
| Offline | C0 | Fine-tuning | 0.0359 ± 0.0121 | 0.0926 ± 0.0139 | 0.8633 ± 0.0145 | 0.0358 | 0.0369 |
| Offline | C1 | Experience replay | 0.3650 ± 0.0187 | 0.5614 ± 0.0193 | 0.2914 ± 0.0327 | 0.1771 | 0.4075 |
| Offline | C2 | DER++ | 0.3565 ± 0.0192 | 0.5949 ± 0.0160 | 0.2373 ± 0.0359 | 0.2238 | 0.3630 |
| Offline | C3 | Replay + distillation | **0.3877 ± 0.0126** | 0.4870 ± 0.0201 | 0.3489 ± 0.0262 | 0.1539 | **0.4928** |
| Offline | C4 | X-DER thích nghi | 0.3723 ± 0.0126 | **0.5969 ± 0.0353** | **0.2324 ± 0.0349** | **0.2195** | 0.4003 |
| Offline | C6 | Perturb-and-Merge thích nghi | 0.0782 ± 0.0298 | 0.1371 ± 0.0413 | 0.3925 ± 0.0843 | 0.0979 | 0.0505 |
| Một lượt | C0O | Fine-tuning một lượt | 0.0072 ± 0.0026 | 0.0209 ± 0.0051 | 0.2890 ± 0.0170 | 0.0225 | 0.0000 |
| Một lượt | C1O | Replay một lượt | **0.2286 ± 0.0090** | **0.4022 ± 0.0301** | **0.0749 ± 0.0079** | **0.1143** | 0.2486 |
| Một lượt | C5 | OTC/MMOT thích nghi | 0.1637 ± 0.0218 | 0.3252 ± 0.0255 | 0.2031 ± 0.0348 | 0.0321 | **0.2494** |

Head/Tail F1 trong bảng là **mean**; độ lệch chuẩn đầy đủ nằm trong CSV nguồn. C3 cao nhất về Macro-F1 trong track offline; C4 tốt hơn về balanced accuracy và forgetting. C5 thấp hơn C1O về Macro-F1 trong track một lượt. C4+GroupPrior ở mục 7 là bước hiệu chỉnh classifier **sau C4**, không phải C4 thô tại đây.

## 6. H15 model merging và chẩn đoán C3/C4

Pilot H15 gồm 6 run T-DDI **seed 0**. `alpha=0.5` cho cả EMA backbone và selective merge. Các dòng cùng P có thể so trực tiếp trong pilot, nhưng một seed không cho phép kết luận đa seed.

| Protocol | Phương án | Macro-F1 ↑ | Balanced acc. ↑ | Forgetting ↓ | Head F1 | Tail F1 |
|---|---|---:|---:|---:|---:|---:|
| P4-H15 | C3, không merge | **0.3474** | 0.4678 | 0.3137 | 0.1115 | **0.4466** |
| P4-H15 | EMA backbone | 0.3062 | **0.4893** | **0.2040** | **0.1372** | 0.3412 |
| P4-H15 | Selective merge | 0.2201 | 0.2282 | 0.4183 | 0.0442 | 0.3374 |
| P9-H15 | C3, không merge | **0.3616** | 0.4734 | 0.2880 | 0.1214 | **0.4602** |
| P9-H15 | EMA backbone | 0.3025 | **0.5199** | **0.1363** | **0.1530** | 0.3188 |
| P9-H15 | Selective merge | 0.1626 | 0.1954 | 0.4851 | 0.0428 | 0.2321 |

EMA giảm forgetting và tăng balanced accuracy/Head F1, đồng thời giảm Macro-F1/Tail F1. Selective merge alpha 0.5 giảm mạnh Macro-F1 trên cả P4 và P9. Nguồn: `outputs/runs_h15_merging/` và bảng trong `CHAT_TRANSCRIPT_P9_H15.md`.

Chẩn đoán plasticity/retention riêng cho C3/C4 ghi các giá trị sau. `New-task F1` là trung bình F1 khi task vừa xuất hiện; `final-task F1` là F1 của các task cũ tại endpoint. Bảng này là chẩn đoán, không thay thế đánh giá endpoint 5 seed ở mục 5.

| Setting | Method | New-task F1 ↑ | Final-task F1 ↑ | Retention ratio ↑ | Forgetting ↓ |
|---|---|---:|---:|---:|---:|
| P4-H8 | C3 | 0.8648 | 0.5530 | 0.6448 | 0.3117 |
| P4-H8 | C4 | 0.8259 | 0.6357 | 0.7757 | 0.1902 |
| P9-H8 | C3 | 0.8638 | 0.5585 | 0.6488 | 0.3053 |
| P9-H8 | C4 | 0.8279 | 0.6246 | 0.7560 | 0.2033 |
| P9-H15 | C3 | 0.8331 | 0.5172 | 0.6351 | 0.3218 |
| P9-H15 | C4 | 0.8208 | 0.6078 | 0.7533 | 0.2193 |

Nguồn: `outputs/runs_c3_c4_diagnostics/analysis/plasticity_retention_summary.csv`. C4 giữ task cũ tốt hơn C3 trong ba setting, còn F1 task mới thấp hơn.

## 7. C4+GroupPrior: xác nhận H8/H15/H29

Candidate `C4-GroupPrior-v1` tại `configs/c4_grouprior_candidate.json` chia lớp theo số mẫu train: Head `>1000`, Medium `101–1000`, Tail `<=100`. Sau mỗi task, tỷ lệ dự đoán và tỷ lệ thật của ba nhóm trên **validation** được dùng để cộng bias vào logit C4 với `lambda=1.0`. Test chỉ dùng để đánh giá. Đây là xác nhận **3 seed 0–2** mỗi setting, chưa phải bảng 5 seed.

| Setting | C3 Macro-F1 | C4 thô Macro-F1 | C4+GroupPrior Macro-F1 | Gain ghép cặp so C3 | Số cặp thắng |
|---|---:|---:|---:|---:|---:|
| P4-H8 | 0.3903 | 0.3712 | 0.4258 | +0.0355 ± 0.0067 | 3/3 |
| P4-H15 | 0.3461 | 0.3518 | 0.4041 | +0.0579 ± 0.0047 | 3/3 |
| P9-H15 | 0.3473 | 0.3325 | 0.3919 | +0.0445 ± 0.0047 | 3/3 |
| P9-H29 | 0.3170 | 0.2763 | 0.3858 | +0.0688 ± 0.0069 | 3/3 |

Các cột Macro-F1 là **mean của đúng ba seed 0–2**. Gain là trung bình chênh lệch từng cặp cùng seed; `±` là standard deviation của chênh lệch ghép cặp. Nguồn H8/H15: `outputs/runs_c4_grouprior_screening/analysis/three_seed_gain_summary.csv`, `three_seed_paired_gains.csv`, `per_task_analysis/final_endpoint_summary.csv`. Nguồn H29: `outputs/runs_c4_grouprior_h29/analysis/h29_three_seed_aggregate.csv` và `h29_three_seed_paired_delta_summary.csv`.

### 7.1 Chi tiết P9-H29, 3 seed

| Method | Macro-F1 ↑ | Balanced acc. ↑ | Head F1 ↑ | Medium F1 ↑ | Tail F1 ↑ | Forgetting ↓ |
|---|---:|---:|---:|---:|---:|---:|
| C3 | 0.3170 ± 0.0189 | 0.4544 ± 0.0087 | 0.1065 ± 0.0020 | 0.3541 ± 0.0181 | 0.3904 ± 0.0343 | 0.2842 ± 0.0182 |
| C4 thô | 0.2763 ± 0.0066 | **0.6110 ± 0.0150** | 0.1914 ± 0.0187 | 0.3636 ± 0.0090 | 0.2584 ± 0.0116 | **0.2461 ± 0.0192** |
| C4+GroupPrior | **0.3858 ± 0.0177** | 0.4800 ± 0.0091 | **0.2003 ± 0.0200** | **0.3971 ± 0.0084** | **0.4645 ± 0.0233** | 0.2525 ± 0.0262 |

C4+GroupPrior hơn C3 ở H29 về Macro-F1 `+0.0688 ± 0.0069`, balanced accuracy `+0.0256 ± 0.0081`, Head F1 `+0.0938 ± 0.0186` và Tail F1 `+0.0741 ± 0.0111`; forgetting giảm trung bình `0.0316` (chênh lệch C4+GroupPrior trừ C3 là `−0.0316 ± 0.0208`). C4 thô có balanced accuracy cao nhất nhưng Macro-F1 thấp hơn C3. Phân tích theo task trên H8/H15 cho thấy hiệu chỉnh cải thiện retention/forgetting nhưng giảm F1 task mới so với C3 và C4 thô. Nguồn: `outputs/runs_c4_grouprior_screening/per_task_analysis/retention_three_seed_summary.csv`; diễn giải: `docs/C4_GROUPPRIOR_PROGRESS.md`.

### 7.2 Chẩn đoán calibration 5 seed, khác study xác nhận

Một chẩn đoán hậu xử lý classifier trên P9-H8/C4 có đủ **5 seed** tại `outputs/runs_c4_classifier_diagnostics/analysis/calibration_test_summary.csv`: C4 thô `0.3723 ± 0.0126` Macro-F1; `group_prior` `0.4271 ± 0.0118`; `class_prior` `0.4164 ± 0.0129`. Đây là phép thử calibration chẩn đoán, không thay cho candidate `C4-GroupPrior-v1` đã khóa và không được gộp với kết quả 3 seed ở bảng trên. File nguồn chứa thêm precision, recall và F1 theo Head/Medium/Tail.

## 8. D1-H29 seed 0 và giới hạn kết luận

- Lịch động `D1-H29` dùng thứ tự lớp ngẫu nhiên, 38 lớp ban đầu và 28 update × 5 lớp. Seed 0 đã hoàn tất: Macro-F1 của C3 là `0.3430`, C4 thô `0.2585`, C4+GroupPrior `0.4239`; gain C4+GroupPrior so với C3 là `+0.0808`. Tail prediction ratio q/p lần lượt là `3.16`, `13.44` và `1.60`. F1 lớp mới của GroupPrior cao hơn C3 `+0.1624`, nhưng forgetting cao hơn `+0.0081` và BWT thấp hơn `0.1389`. Đây là một seed, chưa đủ kết luận về độ bền theo thứ tự lớp. Nguồn: `outputs/runs_c4_grouprior_dynamic/d1/analysis/`.
- P10–P12 và model merging H15 mới có một seed. C4+GroupPrior trên H8/H15/H29 mới được xác nhận với ba seed mỗi setting.
- So sánh H8/H15/H29 chưa kiểm soát tổng compute và tổng replay draws. Cùng số replay mỗi task dẫn tới tổng replay khác nhau khi số task tăng.
- Các metric có thể đổi hướng ưu thế: P2/P3, C3/C4, C4 thô/C4+GroupPrior cho thấy Macro-F1, balanced accuracy, Tail F1 và forgetting không đồng biến. Khi báo cáo kết luận, cần giữ các metric cạnh nhau.
- Artifact run thô trong `outputs/` phần lớn không được Git theo dõi vì kích thước lớn. Tài liệu này ghi đúng trạng thái quan sát tại ngày cập nhật, không tự cập nhật khi run mới hoàn tất.
