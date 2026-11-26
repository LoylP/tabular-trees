# Demo: Why do tree-based models still outperform deep learning on tabular data?

Demo Streamlit cho paper của Grinsztajn, Oyallon & Varoquaux (2022).
Dữ liệu và các phép biến đổi dựa trên repo của tác giả:
https://github.com/LeoGrin/tabular-benchmark (`src/data_transforms.py`, `analyses/*.R`).

## Các tab

| Tab | Nội dung | Phần trong paper |
| --- | --- | --- |
| 1 · Benchmark results | Đường điểm theo số iteration random search, đọc từ file kết quả CSV | Figure 1–2 |
| 2 · Decision boundary | RandomForest vs MLP trên 2 feature quan trọng nhất | Appendix Figure 20 |
| 3 · Finding 1: Smoothing | Làm trơn nhãn train bằng Gaussian kernel smoother | Figure 3 |
| 4 · Finding 2: Uninformative features | Thêm cột nhiễu / bỏ cột ít quan trọng | Figures 4, 5 |
| 5 · Finding 3: Rotation | Xoay ngẫu nhiên dữ liệu | Figure 6 |

## Chạy local

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |  macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Lần đầu chọn một dataset, app tự tải CSV từ Hugging Face
(`inria-soda/tabular-benchmark`, thư mục `clf_num/`) và lưu vào `data/clf_num/`.
Nếu máy không tải được, tải tay file đó và đặt vào `data/clf_num/<tên>.csv`.

### Kết quả benchmark (tab 1)

- Kết quả gốc của tác giả:

  ```bash
  mkdir -p data/results
  curl -L -o data/results/benchmark_total.csv https://figshare.com/ndownloader/files/40081681
  ```

- Kết quả của nhóm: đặt vào `data/results/team_results.csv` (hoặc upload trong app), cùng format:
  `model_name, data__keyword, mean_val_score, mean_test_score`, thêm `hp` (`default` cho lần chạy
  với hyperparameter mặc định) và `benchmark` (ví dụ `numerical_classification_medium`) nếu có.

## Khác biệt so với paper (để ghi trong báo cáo)

- Model: scikit-learn trên CPU (`GradientBoostingClassifier`, `RandomForestClassifier`,
  `MLPClassifier` 4×256). Không có ResNet / FT-Transformer / SAINT, không tune hyperparameter.
- Mỗi lần chạy chỉ dùng 1 dataset và một mẫu con (mặc định 3.000 train / 2.000 test), nên số liệu
  minh họa xu hướng, không thay thế kết quả tổng hợp ở tab 1.
- MLP train bằng Adam nên chỉ bất biến *gần đúng* khi xoay.
- Smoothing: viết lại dạng vectorized cùng công thức với `remove_high_frequency_from_train`.
- Đường random search: port từ code R của tác giả; thứ tự các lần thử ngẫu nhiên được đánh số 1..k
  sau lần chạy default (tác giả để lại một khoảng trống ngẫu nhiên trong thứ tự này).

Repo gốc không có file LICENSE; phần code chuyển thể đã ghi nguồn trong từng file.
