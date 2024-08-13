import streamlit as st
from streamlit_option_menu import option_menu
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import os
label_columns = ['Spiritual', 'Physical', 'Intellectual', 'Social', 'Vocational', 'Emotional']

st.set_page_config(page_title="Mental Health Insights", page_icon=None, layout="wide")
page_bg_img = '''
<style>
.st-emotion-cache-z5fcl4 {padding: 2rem 1rem 4rem}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center'>Mental Health Insights</h1>", unsafe_allow_html=True)
output = './downloaded_file.zip'
model_save_path = './content/saved_model/'
if not os.path.exists(model_save_path):
    import gdown
    import zipfile
    file_id = '1gp1P74uVeRFDuNf5P6-Av-0PjG10HqWA'
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output, quiet=False)
    # 读取压缩文件
    file=zipfile.ZipFile(output)
    file.extractall('./')
    # 关闭文件流
    file.close()
    os.remove(output)

with st.sidebar:
    selected = option_menu("Function selection", ["Application", "Topics", "Sharing"], 
                           icons=['chat-left-text-fill', 'bar-chart-steps', 'share-fill'], menu_icon="robot", default_index=0)


if selected == "Topics":
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["All", "Spiritual", "Physical", "Intellectual", "Social", "Vocational", "Emotional"])
    with tab1:
        with open("All.html", mode='r') as f:
            res = f.read()
        f.close()
        st.components.v1.html(res, height=800, scrolling=True)
    with tab2:
        with open("Spiritual.html", mode='r') as f:
            res = f.read()
        f.close()
        st.components.v1.html(res, height=800, scrolling=True)
    with tab3:
        with open("Physical.html", mode='r') as f:
            res = f.read()
        f.close()
        st.components.v1.html(res, height=800, scrolling=True)
    with tab4:
        with open("Intellectual.html", mode='r') as f:
            res = f.read()
        f.close()
        st.components.v1.html(res, height=800, scrolling=True)
    with tab5:
        with open("Social.html", mode='r') as f:
            res = f.read()
        f.close()
        st.components.v1.html(res, height=800, scrolling=True)
    with tab6:
        with open("Vocational.html", mode='r') as f:
            res = f.read()
        f.close()
        st.components.v1.html(res, height=800, scrolling=True)
    with tab7:
        with open("Emotional.html", mode='r') as f:
            res = f.read()
        f.close()
        st.components.v1.html(res, height=800, scrolling=True)
elif selected == "Application":
    with st.spinner('model load...'):
        # 加载保存的模型和 tokenizer
        loaded_model = AutoModelForSequenceClassification.from_pretrained(model_save_path)
        loaded_tokenizer = AutoTokenizer.from_pretrained(model_save_path)
    col1, col2, col3 = st.columns([2, 6, 2])
    
    with col2:
        st.markdown("<h4 style='font-style: italic'>Please don't worry about our mental health status. Copy and paste your or your friends' posts here so that we can start by identifying specific dimensions and better understand the issues faced by ourselves or others.</h4>", unsafe_allow_html=True)
        inp = st.text_area(
            "Please enter text",
            None,
            height=200
        )
        if st.button("explore", type='primary'):
            inputs = loaded_tokenizer([inp], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
            
            # 模型推理
            outputs = loaded_model(**inputs)
            logits = outputs.logits
            # 通过阈值得到二进制预测
            preds = (logits > 0).int()
            st.write([v for _, v in zip(preds[0].tolist(), label_columns) if _ == 1])
elif selected == 'Sharing':
    
    st.subheader('Categorized URLs', divider='rainbow')
    st.markdown("<h4 style='text-align: center; font-style: italic'>Mental Health and Well-being</h4>", unsafe_allow_html=True)
    col1, col2 = st.columns([5, 5])
    with col1:
        st.page_link("https://www.webmd.com/mental-health/how-does-mental-health-affect-physical-health", 
                     label="*How Does Mental Health Affect Physical Health - WebMD*")
        st.image('./imgs/M1.png')
        st.divider()
        st.page_link("https://www.psychiatry.org/news-room/apa-blogs/purpose-in-life-less-stress-better-mental-health", 
                     label="Purpose in Life Can Lead to Less Stress, Better Mental Well-Being - American\n- Psychiatric Association")
        st.image('./imgs/M4.png')
        st.divider()
        st.page_link("https://mhanational.org/live-your-life-well", 
                     label="Live Your Life Well - Mental Health America")
        st.image('./imgs/M6.png')
        st.divider()
        
        st.page_link("https://www.sciencedaily.com/news/mind_brain/mental_health/", 
                     label="Mental Health News - ScienceDaily")
        st.image('./imgs/M8.png')
        st.divider()
    with col2:
        st.page_link("https://www.cdc.gov/mentalhealth/learn/index.htm", 
                     label="About Mental Health - Centers for Disease Control and Prevention")
        
        st.image('./imgs/M2.png')
        st.divider()
        st.page_link("https://www.psychologytoday.com/us/blog/finding-a-new-home/202302/a-surprising-secret-to-happiness-intense-emotions", 
                     label="A Surprising Secret to Happiness - Psychology Today")
        st.image('./imgs/M3.png')
        st.divider()
        st.page_link("https://childmind.org/article/support-friend-with-mental-health-challenges/", 
                     label="How to Support a Friend with Mental Health Challenges - Child Mind Institute")
        st.image('./imgs/M5.png')
        st.divider()
        
        st.page_link("https://www.nhs.uk/every-mind-matters/lifes-challenges/health-issues/", 
                     label="Health Issues - Every Mind Matters - NHS")
        st.image('./imgs/M7.png')
        st.divider()

    st.markdown("<h4 style='text-align: center; font-style: italic'>Social Relationships and Friendship</h4>", unsafe_allow_html=True)
    col3, col4 = st.columns([5, 5])
    with col3:
        st.page_link("https://www.washingtonpost.com/wellness/2024/05/28/in-person-friendships-health-benefits/", 
                     label="Why In-Person Friendships Are Better for Health than Virtual Pals - Washington Post")
        st.image('./imgs/S1.png')
        st.divider()

    with col4:
        st.page_link("https://www.apa.org/monitor/2023/06/cover-story-science-friendship", 
                     label="The Science of Friendship - American Psychological Association")
        st.image('./imgs/S2.png')
        st.divider()
        st.page_link("https://www.nhs.uk/every-mind-matters/lifes-challenges/maintaining-healthy-relationships-and-mental-wellbeing/", 
                     label="Maintaining Healthy Relationships and Mental Well-Being - NHS")
        st.image('./imgs/S3.png')
        st.divider()
    st.markdown("<h4 style='text-align: center; font-style: italic'>Education and Vocational Aspects</h4>", unsafe_allow_html=True)
    col5, col6 = st.columns([5, 5])
    with col5:
        st.page_link("https://mcc.gse.harvard.edu/reports/on-edge",
                     label="On Edge - Harvard Graduate School of Education")
        st.image('./imgs/E1.png')
        st.divider()
