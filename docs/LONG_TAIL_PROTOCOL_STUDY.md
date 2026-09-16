# P9–P12 Long-Tail Protocol Extension

P9–P12 là study mở rộng tách riêng khỏi protocol study P0–P8. Backbone, continual-learning method, optimizer, replay budget, split và task layout vẫn lấy nguyên từ `configs/protocol_study_tddi.json`; không sửa hoặc trộn kết quả khóa P0–P8.

Pilot hiện tại chạy seed 0 trên T-DDI:

- P9: Tail-Profile Balanced, bốn quantile theo `rank(log(train_count))`.
- P10: Effective-Number Balanced, pilot chính dùng `beta=0.999`; builder hỗ trợ thêm `0.9`, `0.99`, `0.9999`.
- P11: Progressive Tail-Drift, Head `>1000`, Medium `101–1000`, Tail `<=100`, quota nguyên bảo toàn tổng số lớp.
- P12: Tail-Shock, shock ở task 3 (đánh số từ 0), quota mục tiêu `10%/10%/80%`; profile normal được suy ra từ số lớp còn lại để bảo toàn toàn bộ lớp.

Sinh task files:

```bash
.venv/bin/python scripts/build_long_tail_protocols.py \
  --class-counts outputs/class_distribution/class_counts_train.csv \
  --outdir outputs/tasks --protocol all --seeds 0
```

Chạy pilot:

```bash
LONG_TAIL_SEEDS=0 bash scripts/run_long_tail_protocols.sh all
```

Runner mặc định chỉ train P10 với `beta=0.999`. Đổi beta khi đã có task file tương ứng:

```bash
P10_BETA=0.99 LONG_TAIL_SEEDS=0 bash scripts/run_long_tail_protocols.sh P10
```

Metric bổ sung được ghi trong `outputs/runs_long_tail/analysis/`:

- `long_tail_summary.csv`: final Macro-F1, Balanced Accuracy, task forgetting, F1 Head/Medium/Tail, Head–Tail Gap, BWT và variance giữa task.
- `long_tail_metrics_by_task.csv`: trajectory F1 theo nhóm tần suất.
- P12 có thêm shock drop, recovery, tail gain và old-class drop.

Pilot một seed chỉ dùng để kiểm tra pipeline và xu hướng ban đầu. Không dùng nó để tuyên bố thắng/thua cuối cùng; kết luận chính cần đủ 5 training seed như study gốc.

## Tiến độ đến 16/09/2026

- P9 trên T-DDI đã hoàn tất 5 seed 0–4: final Macro-F1 `0.3877 ± 0.0126`, balanced accuracy `0.4870 ± 0.0201`, task forgetting `0.3489 ± 0.0262` (mean ± sample standard deviation). Số liệu từ `outputs/runs_long_tail/analysis_p9_all_seeds/long_tail_aggregate_summary.csv`.
- P9 trên TabM và DDI-GCN đã hoàn tất 5 seed cho mỗi backbone tại `outputs/runs_p9_backbones/`. Phải báo cáo riêng theo backbone.
- P10, P11 và P12 hiện có một run seed 0 cho mỗi protocol trong `outputs/runs_long_tail/`; chưa có đủ 5 seed để kết luận đa seed.
