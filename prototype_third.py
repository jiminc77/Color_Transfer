import streamlit as st
from PIL import Image
from instagrapi import Client
from pathlib import Path
import requests
import os
import subprocess
import shutil
import time
import sys
sys.path.append('/home/jovyan/Color_Transfer/color-transfer')
from color_transfer import run

streamlit_style = """
			<style>
			@import url("https://fonts.googleapis.com/css2?family=Poppins&display=swap");

			html, body, [class*="css"]  {
			font-family: 'Poppins', sans-serif;
			}
            .custom-style {
                margin-top: 115px;  # 이미지 사이의 간격 조절
            }
            .custom-title {
                font-weight: 900;  # 텍스트를 굵게
            }
            </style>
			"""

st.markdown(streamlit_style, unsafe_allow_html=True)

def display_available_memory():
    mem_info = psutil.virtual_memory()
    available_memory = mem_info.available / (1 << 20)  # Convert to MB
    st.write(f"Available memory: {available_memory:.2f} MB")

def insta_crawling(ID, PW):
    # jaeu8021
    # kvoid2824#
    
    cl = Client()
    try:
        cl.login('jaeu8021', 'kvoid2824#')
    except Exception as e:
        st.write("로그인 중 에러 발생:", e)

    st.write("Sucesses")

    user_id = cl.user_id_from_username("jaeu8021")
    state_text.text("Feed searching...")

    medias = cl.user_medias(int(user_id), 9)

    folder = "test-folder"
    createDirectory(folder)
    state_text.text("Saving Image....")
    temp = []
    for m in medias:
        try:
            p = photo_download(cl, m.pk, folder)
            temp.append(p)
        except AssertionError:
            pass
    
    crawled = temp[::]
    # state_text.text("Crawling finished! " + os.path.abspath(p))
    # st.image(Image.open(os.path.abspath(p)))


def photo_download(c, pk, folder):
    media = c.media_info(pk)
    assert media.media_type == 1, "Must been photo"
    filename = "{username}_{media_pk}".format(
        username=media.user.username, media_pk=pk
    )

    p = os.path.join(folder, filename + '.jpg')
    response = requests.get(media.thumbnail_url,
                            stream=True, timeout=c.request_timeout)
    response.raise_for_status()
    with open(p, "wb") as f:
        f.write(response.content)

    return p


def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")


def concat_image(files, progress_callback):  # test folder 에서 이미지를 받아와서 합해야됨

    def resize_squared_img(img):
        h = img.height
        w = img.width
        if w < h:
            m = (h-w)//2
            return img.crop((0, m, w, m+w)), w
        elif h < w:
            m = (w-h)//2
            return img.crop((m, 0, m+h, h)), h
        return img, h

    images = []
    msize = 200

    for f in files:
        img = f
        img, m = resize_squared_img(img)
        msize = min(m, msize)
        images.append(img)

    def hconcat_resize_pil(im_list):
        msize = 200
        im_list_resize = [im.resize((msize, msize))
                          for im in im_list]
        total_width = msize*len(im_list)
        dst = Image.new('RGB', (total_width, msize))
        pos_x = 0
        for im in im_list_resize:
            dst.paste(im, (pos_x, 0))
            pos_x += msize
        return dst

    def vconcat_pil(im_list):
        msize = 200
        total_height = msize*len(im_list)
        dst = Image.new('RGB', (msize*3, total_height))
        pos_y = 0
        for im in im_list:
            dst.paste(im, (0, pos_y))
            pos_y += msize
        return dst

    concat_row = []
    n = len(images)
    for i in range(0, n, 3):
        if n-i < 3:
            break
        row = hconcat_resize_pil(images[i:i+3])
        concat_row.append(row)
        progress = (i + 3) / n
        progress_callback(progress)

    concat_single_image = vconcat_pil(concat_row)
    # st.image(concat_single_image)

    createDirectory('examples/style')
    createDirectory('examples/style_segment')
    createDirectory('examples/content')
    createDirectory('examples/content_segment')
    shutil.copyfile('black_.png', 'examples/style_segment/black_.png')

    concat_single_image.save('./examples/style/concat_image.jpg', 'JPEG')

def update_progress_bar(progress):
    
    if progress < 0.99:
        bar.progress(progress)
    else:
        bar.progress(progress)
        time.sleep(1)
        bar.empty()
        

def delete_all_files(filepath):
    if os.path.exists(filepath):
        for file in os.scandir(filepath):
            os.remove(file.path)
        return "Remove All File"
    else:
        return "Directory Not Found"

def memory_usage(message):
    p = psutil.Process()
    rss = p.memory_info().rss/2**20
    st.write(f"[{message}] memory usage: {rss:10.5f} MB")

def is_square(image):
    width, height = image.size
    return width == height

st.image("intersection.png", width = 100)
st.markdown('<h1 class="custom-title">AI Color Grader</h1>', unsafe_allow_html=True)
st.subheader('Find the filter that best fits your Instagram feed!')

with st.container():
    col1, col2 = st.columns(2)

    with col1:
        target_file = st.file_uploader(label="Choose an image to apply color correction",
                                       type=['jpeg', 'png', 'jpg', 'heic'],
                                       label_visibility='visible',
                                       accept_multiple_files=False)
        if target_file:
            target_image = Image.open(target_file)
            
    with col2:
        uploaded_files = st.file_uploader(label="Choose image(s) for AI to analyze",
                                          type=['jpeg', 'png', 'jpg', 'heic'],
                                          label_visibility='visible',
                                          accept_multiple_files=True)
        pass

    
if target_file:
    target = Image.open(target_file)
    with col1:
        st.markdown('<div class="custom-style"></div>', unsafe_allow_html=True)
        st.image(target)

crawled = []

if 'process_idx' not in st.session_state:
    st.session_state.process_idx = 1

# Check if the user has uploaded any files
if uploaded_files or crawled:
    # if uploaded_files or crawled:

    images = crawled[::]
    # Create an empty list to store the images

    # Loop through each uploaded file and append the opened image to the list
    for file in uploaded_files:
        image = Image.open(file)
        images.append(image)
    
    if st.session_state.process_idx == 1:
        if st.button("Process Images"):
            delete_all_files('examples/content')
            delete_all_files('outputs')

            bar = st.progress(0)
            single = concat_image(images, update_progress_bar)
            target.save('./examples/content/target.jpg', 'JPEG')
            concat = Image.open('./examples/style/concat_image.jpg')
            col2.image(concat)
            st.session_state.process_idx = 2

    if st.session_state.process_idx == 2:
        if st.button("Start Transfer"):   
            directory = './outputs'
            run()
            concat = Image.open('./examples/style/concat_image.jpg')
            col2.image(concat)
            with st.container():
                st.image('./outputs/output.png', use_column_width=True)
                st.session_state.process_idx = 3


    if st.session_state.process_idx == 3:
        with open('./outputs/output.png', 'rb') as file:
            button = st.download_button(label = 'Download', data = file, file_name = "Color_Grading.jpg", mime = 'image/jpg')

    


#id = "leessunj"
#pwd = "Ilsj08282!"

# 서버가 종료되지 않았다면, netstat -lnp | grep [포트번호] 후, kill -9 [process_id]

