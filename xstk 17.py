import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import geostatspy.geostats as geostats
import geostatspy.GSLIB as GSLIB

# 1. Cấu hình giao diện chuẩn 
st.set_page_config(page_title="Phần mềm Dự báo Trữ lượng - Nhóm 1", layout="wide")

st.title("Hệ thống Phân tích Địa thống kê & Quản trị Rủi ro Trữ lượng")
st.markdown("""
*Ứng dụng Khoa học dữ liệu không gian để xác định các mốc rủi ro P10, P50, P90 theo chuẩn Probabilistic Mineral Resource Estimation.*
""")
st.divider()

# 2. Sidebar - Thay đổi trực tiếp số liệu
st.sidebar.header("Bảng điều khiển thông số")
uploaded_file = st.sidebar.file_uploader("Tải lên dữ liệu khoan (.csv)", type=["csv"])

st.sidebar.subheader("Tham số mô phỏng Monte Carlo")
area = st.sidebar.slider("Diện tích mỏ (m2)", 500000, 5000000, 1000000)
thickness = st.sidebar.slider("Độ dày vỉa trung bình (m)", 5, 50, 15)
density = st.sidebar.slider("Tỷ trọng quặng (tấn/m3)", 2.0, 3.5, 2.5)
n_sim = st.sidebar.number_input("Số lần mô phỏng (L)", value=10000, step=1000)

# 3. Xử lý dữ liệu đầu vào
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    # Dữ liệu giả lập nếu chưa có file
    np.random.seed(42)
    df = pd.DataFrame({
        'X': np.random.uniform(0, 1000, 100),
        'Y': np.random.uniform(0, 1000, 100),
        'Porosity': np.random.lognormal(mean=2.5, sigma=0.4, size=100)
    })
    st.info("💡 Mẹo: Bạn có thể thay đổi các thanh trượt ở bên trái để thấy biểu đồ cập nhật ngay lập tức!")

# 4. Phân tích Địa thống kê (Geostatspy)
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Khử cụm & Hiệu chỉnh mẫu")
    
   # ---  CHO PHÉP CHỌN  POROSITY VÀ PERMEABILITY ---
    # Chỉ giữ lại các cột liên quan đến thuộc tính địa chất
    target_keywords = ['poro', 'perm']
    available_cols = [c for c in df.columns if any(k in c.lower() for k in target_keywords)]
    
    if not available_cols:
        # Nếu không tìm thấy từ khóa, hiển thị các cột số trừ X, Y
        available_cols = [c for c in df.columns if c not in ['X', 'Y', 'Weights', 'Well_ID', 'Well ID']]

    vcol = st.selectbox("Chọn thuộc tính phân tích (Porosity hoặc Permeability):", available_cols)
    # Thực hiện khử cụm
    wts, csize, n_eff = geostats.declus(df, 'X', 'Y', vcol, iminmax=1, noff=10, ncell=100, cmin=10, cmax=1000)
    df['Weights'] = wts
    
    raw_mean = df[vcol].mean()
    declust_mean = np.average(df[vcol], weights=df['Weights'])
    
    st.write("---")
    st.metric("Trung bình sau khử cụm", f"{declust_mean:.2f}", f"{declust_mean - raw_mean:.2f} so với thô")
    st.caption("Việc khử cụm giúp loại bỏ sai số do lấy mẫu ưu tiên vào vùng giàu quặng.")

with col2:
    st.subheader(" Bản đồ phân bố không gian")
    fig_map, ax_map = plt.subplots(figsize=(5, 4))
    GSLIB.locmap_st(df, 'X', 'Y', vcol, 0, 1000, 0, 1000, df[vcol].min(), df[vcol].max(), 
                    "Vị trí các lỗ khoan", "Kinh độ (X)", "Vĩ độ (Y)", vcol, plt.cm.plasma)
    st.pyplot(fig_map)
# ---  KHỚP PHÂN PHỐI & KIỂM CHỨNG  ---
st.divider()
st.subheader(" Khớp phân phối & Kiểm chứng dữ liệu")
c_plot1, c_plot2 = st.columns(2)

with c_plot1:
    st.markdown("**Biểu đồ Histogram (Khớp phân phối Gaussian)**")
    fig_hist, ax_hist = plt.subplots()
    sns.histplot(df[vcol], kde=True, color="skyblue", ax=ax_hist)
    ax_hist.set_title(f"Phân phối thực tế của {vcol}")
    st.pyplot(fig_hist)
    st.caption("Đường cong KDE giúp xác định hình dạng phân phối (Gaussian/Lognormal).")

with c_plot2:
    st.markdown("**Biểu đồ Q-Q Plot (Kiểm chứng chuẩn hóa)**")
    fig_qq, ax_qq = plt.subplots()
    # Chuyển đổi nscore để đưa dữ liệu về không gian Gaussian chuẩn hóa
    stats.probplot(df[vcol], dist="norm", plot=plt)
    ax_qq.set_title("Quantile-Quantile Plot")
    st.pyplot(fig_qq)

# 5. Mô phỏng Monte Carlo và Biểu đồ chi tiết
st.divider()
st.subheader(" Dự báo rủi ro & Xác định P-Values")

# Thực hiện tính toán trữ lượng
sim_values = np.random.normal(declust_mean, df[vcol].std(), n_sim)
# Công thức: Trữ lượng = Diện tích * Dày * (Hàm lượng/100) * Tỷ trọng
reserves = area * thickness * (sim_values / 100) * density
reserves = reserves[reserves > 0]

# Tính toán các mốc P
p90 = np.percentile(reserves, 10) # 90% cơ hội đạt mức này hoặc cao hơn
p50 = np.percentile(reserves, 50) # Trung vị
p10 = np.percentile(reserves, 90) # 10% cơ hội đạt mức cực cao

# Hiển thị chỉ số
c1, c2, c3 = st.columns(3)
with c1:
    st.warning(f"**P90 (An toàn):** \n\n {p90:,.0f} Tấn")
with c2:
    st.success(f"**P50 (Cơ sở):** \n\n {p50:,.0f} Tấn")
with c3:
    st.error(f"**P10 (Lạc quan):** \n\n {p10:,.0f} Tấn")

# Vẽ biểu đồ CDF chi tiết
fig_cdf, ax_cdf = plt.subplots(figsize=(10, 6))
sns.ecdfplot(reserves, ax=ax_cdf, color='darkred', linewidth=2.5, label='Đường cong tích lũy (CDF)')

# Thêm các đường gióng P-values
ax_cdf.axvline(p90, color='blue', linestyle='--', alpha=0.6)
ax_cdf.text(p90, 0.1, f'  P90: {p90:,.0f}', color='blue', fontweight='bold')

ax_cdf.axvline(p50, color='green', linestyle='--', alpha=0.6)
ax_cdf.text(p50, 0.5, f'  P50: {p50:,.0f}', color='green', fontweight='bold')

ax_cdf.axvline(p10, color='orange', linestyle='--', alpha=0.6)
ax_cdf.text(p10, 0.9, f'  P10: {p10:,.0f}', color='orange', fontweight='bold')

ax_cdf.set_title("Biểu đồ Phân phối Tích lũy Trữ lượng Dự báo", fontsize=14)
ax_cdf.set_xlabel("Trữ lượng Khoáng sản (Tấn)", fontsize=12)
ax_cdf.set_ylabel("Xác suất tích lũy (Cumulative Probability)", fontsize=12)
ax_cdf.grid(True, alpha=0.3)
ax_cdf.legend()

st.pyplot(fig_cdf)

# 6. Giải thích kết quả cho báo cáo
st.info(f"""
**Giải thích chuyên sâu cho báo cáo[cite: 81, 91]:**
1. **Đường cong CDF:** Cho thấy toàn bộ dải bất định của vỉa. Độ dốc của đường cong thể hiện mức độ rủi ro (đường càng dốc, rủi ro càng thấp).
2. **Mốc P90 ({p90:,.0f} tấn):** Là mức trữ lượng mà nhà đầu tư có thể tin tưởng với độ tin cậy 90%. Đây là con số dùng để chứng minh năng lực tài chính với ngân hàng[cite: 39, 83].
3. **Mốc P10 ({p10:,.0f} tấn):** Đại diện cho các "vỉa giàu" tiềm năng chưa được khám phá hết, giúp kỳ vọng vào lợi nhuận đột biến[cite: 86].
""")