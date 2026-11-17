# Tiêu chí đặt trước cho bước 2a và 2b (ghi ngày 2026-10-15, trước khi chạy)

## 2a. EvPFC trên không gian ẩn tiền huấn luyện của scPIKA (encoder `ae`)

- Không gian ẩn: giai đoạn tiền huấn luyện của bộ tự mã hóa scPIKA, 30 epoch, chỉ dùng hàm mất mát tái tạo, không dùng nhãn. Seed 3–7.
- Dữ liệu: BMNC, Ma2020_1, PBMC10k_Multiome_feat, SMAGE3K (đã xử lý), và SLN111, SLN206, CellMix, ASAP_PBMC (có số đếm).
- So sánh với: EvPFC trên PCA (R7), scPIKA với bộ mã hóa riêng (R1, cùng seed và cùng tập nhãn), eKNN và LS trên cùng không gian ẩn.

| Mã | Tiêu chí | Ngưỡng |
|---|---|---|
| A1 | Trung bình ARI trên 4 bộ đã xử lý, 5% nhãn: EvPFC-ae so với scPIKA (bộ mã hóa riêng) | EvPFC-ae ≥ scPIKA − 0,01 |
| A2 | Như A1 nhưng so với EvPFC trên PCA | EvPFC-ae > EvPFC-PCA |
| A3 | 20% nhãn nhiễu: EvPFC-ae so với scPIKA (bộ mã hóa riêng) trên các bộ có trong R2 | EvPFC-ae > scPIKA |

## 2b. Cụm tự do với ngưỡng conformal

**Cơ chế:**
1. Giữ lại 20% tế bào có nhãn làm tập hiệu chỉnh.
2. Dựng tri thức evidential k-NN chỉ từ 80% nhãn còn lại.
3. Ngưỡng q là phân vị conformal mức 1 − α (α = 0,05) của m(Θ) trên tập hiệu chỉnh.
4. Tế bào không có nhãn có m(Θ) > q được coi là "kiểu mới" và phân vào K − T cụm tự do bằng PFC không có tiên nghiệm.
5. Các tế bào còn lại được gán bằng EvPFC trên T cụm đã biết, dùng tất cả nhãn.

Vì nhãn được lấy ngẫu nhiên, tế bào hiệu chỉnh hoán đổi được với tế bào chưa có nhãn thuộc kiểu đã biết. Do đó tỉ lệ tế bào kiểu đã biết bị đưa nhầm vào cụm tự do có kỳ vọng ≤ α.

**Đánh giá:** ẩn 2 kiểu, 5% nhãn, **seed 8–12**. Đây là các lần lấy nhãn và kiểu ẩn mới, chưa dùng ở bất kỳ bước nào trước đó. Không gian ẩn dùng lại của seed 3–7 (không gian ẩn không dùng nhãn). Dữ liệu: các bộ có ≥ 6 kiểu tế bào.

| Mã | Tiêu chí | Ngưỡng |
|---|---|---|
| B1 | Tỉ lệ tế bào kiểu ẩn vào cụm tự do (trung bình các bộ) | > 30% |
| B2 | ARI so với EvPFC không có ngưỡng conformal | ≥ EvPFC − 0,01 |
| B3 | Tỉ lệ tế bào kiểu đã biết bị đưa vào cụm tự do (trung bình) | ≤ 0,07 (α + sai số) |
| B4 | Báo cáo bắt buộc (không phải tiêu chí): ARI khi bật cơ chế này nhưng **không** có kiểu mới (5% nhãn, K = T + 2) — chi phí của báo động nhầm | — |

**Bổ sung (ghi trước khi chạy 2b):**
- Không gian ẩn: totalVI cho SLN111 và SLN206. Với các bộ đã xử lý, chạy và báo cáo **cả** PCA lẫn `ae`.
- Tiêu chí B1–B3 tính trên **trung bình tất cả các cặp (bộ dữ liệu, không gian ẩn)**.
