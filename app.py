import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from arima_model import run_arima, forecast_next_month_arima
from lstm_model import run_lstm, forecast_next_month_lstm

st.title("📊 Dự báo sản lượng xăng dầu")

# ===== UPLOAD NHIỀU FILE =====
files = st.file_uploader(
    "📁 Upload nhiều file Excel",
    type=["xlsx"],
    accept_multiple_files=True
)

if not files:
    st.warning("Vui lòng upload file")
    st.stop()

# ===== GỘP FILE =====
df_list = []

for file in files:
    try:
        temp = pd.read_excel(file)
        df_list.append(temp)
    except:
        st.warning(f"Lỗi file: {file.name}")

df = pd.concat(df_list, ignore_index=True)

st.success(f"✅ Đã tải {len(files)} file")

# ===== CHUẨN HÓA =====
df.columns = df.columns.str.strip()

# tìm cột ngày
for c in df.columns:
    if "ngày" in c.lower():
        df.rename(columns={c: "Ngày"}, inplace=True)

# tìm cột sản lượng
for c in df.columns:
    if "xăng" in c.lower() or "lượng" in c.lower():
        df.rename(columns={c: "Sản lượng xăng"}, inplace=True)

st.subheader("📊 Dữ liệu")
st.dataframe(df.head())

# ===== CHỌN MODE =====
mode = st.radio("Chọn chức năng", ["So sánh mô hình", "Dự báo tháng tới"])

# =========================================
# 📊 SO SÁNH
# =========================================
if mode == "So sánh mô hình":

    train_a, test_a, forecast_a, rmse_a = run_arima(df)
    data_l, pred_l, actual_l, rmse_l = run_lstm(df)

    st.subheader("📊 Kết quả")

    st.write(f"ARIMA RMSE: {rmse_a}")
    st.write(f"LSTM RMSE: {rmse_l}")

    fig, ax = plt.subplots()

    ax.plot(test_a.index, test_a.values, label="Thực tế")
    ax.plot(test_a.index, forecast_a, label="ARIMA")

    ax.legend()
    st.pyplot(fig)

# =========================================
# 🚀 DỰ BÁO
# =========================================
if mode == "Dự báo tháng tới":

    steps = st.number_input("Số ngày dự báo", 7, 60, 30)

    data_a, future_a = forecast_next_month_arima(df, steps)
    data_l, future_l = forecast_next_month_lstm(df, steps)

    future_dates = pd.date_range(start=data_a.index[-1], periods=steps+1)[1:]

    result = pd.DataFrame({
        "Ngày": future_dates,
        "ARIMA": future_a,
        "LSTM": future_l
    })

    st.subheader("📈 Kết quả dự báo")
    st.dataframe(result)

    fig, ax = plt.subplots()

    ax.plot(data_a.index, data_a.values, label="Lịch sử")
    ax.plot(future_dates, future_a, label="ARIMA", linestyle="--")
    ax.plot(future_dates, future_l, label="LSTM", linestyle="--")

    ax.legend()
    st.pyplot(fig)