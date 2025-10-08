import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError
from google.genai import types # Thêm import types

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài Chính & Chatbot 📊")

# --- Khởi tạo Lịch sử Chat và Session (chỉ chạy một lần) ---
# Lịch sử tin nhắn
if "messages" not in st.session_state:
    st.session_state.messages = []
# Phiên chat để duy trì ngữ cảnh hội thoại
if "chat_session" not in st.session_state:
    st.session_state.chat_session = None

# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 1. Tính Tốc độ Tăng trưởng
    # Dùng .replace(0, 1e-9) cho Series Pandas để tránh lỗi chia cho 0
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    # Lọc chỉ tiêu "TỔNG CỘNG TÀI SẢN"
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    # Xử lý giá trị 0 cho mẫu số
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    # Tính tỷ trọng với mẫu số đã được xử lý
    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    
    return df

# --- Hàm gọi API Gemini để phân tích báo cáo ---
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash' 

        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành.
        
        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except KeyError:
        return "Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'. Vui lòng kiểm tra cấu hình Secrets trên Streamlit Cloud."
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

# --- Hàm khởi tạo Chat Session ---
def setup_chat_session(api_key):
    """Khởi tạo hoặc kiểm tra phiên chat của Gemini."""
    if st.session_state.chat_session is None:
        try:
            # Khởi tạo client 
            # LƯU Ý: client được khởi tạo lại trong handle_chat_input để tránh bị đóng
            client = genai.Client(api_key=api_key)
            
            # Hệ thống hướng dẫn cho Chatbot
            system_instruction = "Bạn là một trợ lý tài chính thân thiện và chuyên nghiệp. Hãy trả lời các câu hỏi về tài chính, kinh doanh, và các chỉ số tài chính một cách rõ ràng và dễ hiểu."
            
            # Truyền system instruction qua GenerateContentConfig
            config = types.GenerateContentConfig(
                system_instruction=system_instruction
            )
            
            st.session_state.chat_session = client.chats.create(
                model="gemini-2.5-flash",
                config=config # Truyền config vào đây
            )
            # Thêm tin nhắn chào mừng ban đầu
            st.session_state.messages.append(
                {"role": "assistant", "content": "Xin chào! Tôi là trợ lý tài chính AI của bạn. Hãy hỏi tôi bất cứ điều gì về phân tích tài chính hoặc các chỉ số bạn quan tâm."}
            )
            
        except Exception as e:
            st.error(f"Lỗi khởi tạo Chat Session: Vui lòng kiểm tra Khóa API. Chi tiết lỗi: {e}")
            return False
    return True

# --- Hàm xử lý Input Chat ---
def handle_chat_input(prompt):
    """Gửi prompt đến Gemini Chat Session và cập nhật lịch sử."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Khởi tạo lại client để tránh lỗi "client has been closed" do rerun Streamlit
    api_key_chat = st.secrets.get("GEMINI_API_KEY")
    if not api_key_chat:
        st.session_state.messages.append({"role": "assistant", "content": "Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'."})
        return

    try:
        # Khởi tạo client
        client = genai.Client(api_key=api_key_chat)
        
        # Gửi tin nhắn và nhận phản hồi từ phiên chat đã có ngữ cảnh
        response = st.session_state.chat_session.send_message(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response.text})
    except Exception as e:
        st.session_state.messages.append({"role": "assistant", "content": f"Đã xảy ra lỗi trong quá trình chat: {e}"})


# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

# --- Logic chính khi có file được tải lên ---
if uploaded_file is not None:
    try:
        df_raw = pd.read_excel(uploaded_file)
        
        # Tiền xử lý: Đảm bảo chỉ có 3 cột quan trọng
        df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
        
        # Xử lý dữ liệu
        df_processed = process_financial_data(df_raw.copy())

        if df_processed is not None:
            
            # --- Chức năng 2 & 3: Hiển thị Kết quả ---
            st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
            st.dataframe(df_processed.style.format({
                'Năm trước': '{:,.0f}',
                'Năm sau': '{:,.0f}',
                'Tốc độ tăng trưởng (%)': '{:.2f}%',
                'Tỷ trọng Năm trước (%)': '{:.2f}%',
                'Tỷ trọng Năm sau (%)': '{:.2f}%'
            }), use_container_width=True)
            
            # --- Chức năng 4: Tính Chỉ số Tài chính ---
            st.subheader("4. Các Chỉ số Tài chính Cơ bản")
            
            # Khởi tạo giá trị mặc định cho chỉ số thanh toán
            thanh_toan_hien_hanh_N = "N/A"
            thanh_toan_hien_hanh_N_1 = "N/A"

            try:
                # Lấy Tài sản ngắn hạn
                tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Lấy Nợ ngắn hạn
                no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]  
                no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Tránh chia cho 0
                if no_ngan_han_N != 0:
                    thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N
                if no_ngan_han_N_1 != 0:
                    thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1
                
                # Hiển thị chỉ số
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm trước)",
                        value=f"{thanh_toan_hien_hanh_N_1:.2f} lần" if isinstance(thanh_toan_hien_hanh_N_1, (int, float)) else thanh_toan_hien_hanh_N_1
                    )
                with col2:
                    delta_value = None
                    if isinstance(thanh_toan_hien_hanh_N, (int, float)) and isinstance(thanh_toan_hien_hanh_N_1, (int, float)):
                        delta_value = f"{thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1:.2f}"
                        
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm sau)",
                        value=f"{thanh_toan_hien_hanh_N:.2f} lần" if isinstance(thanh_toan_hien_hanh_N, (int, float)) else thanh_toan_hien_hanh_N,
                        delta=delta_value
                    )
                    
            except IndexError:
                st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số.")
                
            # --- Chức năng 5: Nhận xét AI ---
            st.subheader("5. Nhận xét Tình hình Tài chính (AI)")
            
            # Chuẩn bị dữ liệu để gửi cho AI
            data_for_ai = pd.DataFrame({
                'Chỉ tiêu': [
                    'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                    'Tăng trưởng Tài sản ngắn hạn (%)', 
                    'Thanh toán hiện hành (N-1)', 
                    'Thanh toán hiện hành (N)'
                ],
                'Giá trị': [
                    df_processed.to_markdown(index=False),
                    f"{df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Tốc độ tăng trưởng (%)'].iloc[0]:.2f}%", 
                    f"{thanh_toan_hien_hanh_N_1}", 
                    f"{thanh_toan_hien_hanh_N}"
                ]
            }).to_markdown(index=False) 

            if st.button("Yêu cầu AI Phân tích"):
                api_key_analysis = st.secrets.get("GEMINI_API_KEY") 
                
                if api_key_analysis:
                    with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                        ai_result = get_ai_analysis(data_for_ai, api_key_analysis)
                        st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                        st.info(ai_result)
                else:
                    st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")

    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")

else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")

# -------------------------------------------------------------
# --- Chức năng MỚI: Hộp thoại Hỏi đáp Tài chính (Chatbot) ---
# -------------------------------------------------------------
st.markdown("---")
st.subheader("6. Trợ lý Hỏi đáp Tài chính AI (Chatbot) 💬")

api_key_chat = st.secrets.get("GEMINI_API_KEY")

if api_key_chat:
    # 1. Khởi tạo/kiểm tra phiên chat session
    if setup_chat_session(api_key_chat):
        # 2. Hiển thị lịch sử tin nhắn
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 3. Xử lý input mới
        if prompt := st.chat_input("Hỏi tôi về các chỉ số tài chính, phân tích Bảng Cân đối Kế toán, hoặc các khái niệm liên quan..."):
            handle_chat_input(prompt)
            # Rerun để hiển thị tin nhắn mới ngay lập tức
            st.rerun() 
else:
    st.warning("Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets để sử dụng Trợ lý AI.")
