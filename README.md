📈 Advanced Stock Analysis Dashboard (Bảng thông tin Phân tích Cổ phiếu Nâng cao)

Dự án này là một ứng dụng web tương tác được xây dựng bằng Streamlit, cung cấp một công cụ toàn diện để phân tích thị trường chứng khoán Việt Nam (qua vnstock) và thị trường quốc tế (qua yfinance).

Ứng dụng giúp người dùng đưa ra quyết định giao dịch dựa trên sự kết hợp mạnh mẽ giữa phân tích kỹ thuật, phân tích cơ bản và mô hình dự đoán.

requirements.txt

streamlit
yfinance
pandas
pandas-ta
plotly
numpy
vnstock
requests
pmdarima
statsmodels
scikit-learn
prophet
beautifulsoup4
vaderSentiment

streamlit run k.py

Thao tác Cơ bản

Nhập Mã: Nhập mã cổ phiếu đơn lẻ (ví dụ: NVDA, FPT) hoặc nhiều mã để so sánh (ví dụ: FPT, VCB, HPG).

Chọn Phạm vi: Chọn phạm vi thời gian (ví dụ: 3M, 1Y).

Nhấn "Phân tích": Ứng dụng sẽ tải dữ liệu, tính toán các chỉ báo và cập nhật tất cả các tab.
