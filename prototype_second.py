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
from transfer import run

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
    available_memory = mem_info.available / (1024 ** 2)  # Convert to MB
    st.write(f"Available memory: {available_memory:.2f} MB")

def insta_crawling(ID, PW,target="jaeu8021"):
    cl = Client()
    cl.login(ID, PW)
    print(">>>>>",target)
    user_id = cl.user_id_from_username(target)
    st.text("Feed searching...")

    medias = cl.user_medias_v1(int(user_id), 9)

    folder = "test-folder"
    createDirectory(folder)
    # delete_all_files('test-folder')
    st.text("Saving Image....")
    temp = []
    for m in medias:
        try:
            p = photo_download(cl, m.pk, folder)
            temp.append(p)
        except AssertionError:
            pass
    st.text("Crawling finished! " + os.path.abspath(p))
    st.image(Image.open(os.path.abspath(p)))

    st.session_state.crawled=temp[::]


def photo_download(c, pk, folder):
    media = c.media_info(pk)
    
    filename = "{username}_{media_pk}".format(
        username=media.user.username, media_pk=pk
    )
    p = os.path.join(folder, filename + '.jpg')
    print("INFO", media.media_type)
    if media.media_type==8:
        response = requests.get(media.resources[0].thumbnail_url,
                            stream=True, timeout=c.request_timeout)
    else:
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
    print("start concating...")
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
    msize = 1000

    for f in files:
        img = f
        img, m = resize_squared_img(img)
        msize = min(m, msize)
        images.append(img)

    def hconcat_resize_pil(im_list,msize):
        im_list_resize = [im.resize((msize, msize))
                          for im in im_list]
        total_width = msize*len(im_list)
        dst = Image.new('RGB', (total_width, msize))
        pos_x = 0
        for im in im_list_resize:
            dst.paste(im, (pos_x, 0))
            pos_x += msize
        return dst

    def vconcat_pil(im_list,msize):
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
        row = hconcat_resize_pil(images[i:i+3],msize)
        concat_row.append(row)
        progress = (i + 3) / n
        progress_callback(progress)

    concat_single_image = vconcat_pil(concat_row,msize)
    # st.image(concat_single_image)
    createDirectory('examples/style')
    createDirectory('examples/style_segment')
    createDirectory('examples/content')
    createDirectory('examples/content_segment')
    shutil.copyfile('black_.png', 'examples/style_segment/black_.png')
    shutil.copyfile('black_.png', 'examples/content_segment/black_.png')

    concat_single_image.save('examples/style/concat_image.jpg', 'JPEG')

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
        
            if not is_square(target_image):
                st.error("Please upload a square image.")
            else:
                pass

    with col2:
        st.session_state.uploaded = st.file_uploader(label="Choose image(s) for AI to analyze",
                                          type=['jpeg', 'png', 'jpg', 'heic'],
                                          label_visibility='visible',
                                          accept_multiple_files=True)
        st.text("If you get images by instagram login,\n Submit the form!")
        with st.form("crawling"):
            insta_id = st.text_input("Put your Instagram ID here!")
            insta_pwd = st.text_input('Put your Instagram password here!')
            # Instagram crawling button
            
            username = st.text_input("Put target Instagram ID here if you want!",placeholder="default:your_id",value=insta_id)
            
            submitted = st.form_submit_button("Submit")
            if submitted:
                if not username:
                    username=insta_id
                st.write("Crawling photos from ",username)
                insta_crawling(insta_id, insta_pwd,target=username)
                st.session_state.process_idx = 2
    
           
if target_file:
    target = Image.open(target_file)
    with col1:
        st.markdown('<div class="custom-style"></div>', unsafe_allow_html=True)
        st.image(target)

if 'process_idx' not in st.session_state:
    st.session_state.process_idx = 1
if 'crawled' not in st.session_state:
    st.session_state.crawled=[]
if 'uploaded' not in st.session_state:
    st.session_state.uploaded=[]
if 'images' not in st.session_state:
    st.session_state.images=[]
if 'ref' not in st.session_state:
    st.session_state.ref=0
    
# Check if the user has uploaded any files


def get_images(li):
    imgs=[]
    for file in li:
        image = Image.open(file)
        imgs.append(image)
    return imgs

if st.session_state.crawled:
    st.session_state.images=get_images(st.session_state.crawled)
    col2.markdown("**reference images from CRAWLING**")
if st.session_state.uploaded:
    st.session_state.images=get_images(st.session_state.uploaded)
    col2.markdown("**reference images from uploading**")
    if st.session_state.process_idx<2:
        st.session_state.process_idx = 2
    
    
with col2:
    if st.session_state.process_idx == 2:
        if st.button("Process Images"):
            images=st.session_state.images
            delete_all_files('examples/content')
            delete_all_files('outputs')
            print("processing!!!")
            bar = st.progress(0)
            single = concat_image(images, update_progress_bar)
            target.save('examples/content/target.jpg', 'JPEG')
            concat = Image.open('examples/style/concat_image.jpg')
            col2.image(concat)
            st.session_state.process_idx = 3
        
st.write(st.session_state.process_idx)

if st.session_state.process_idx == 3:
    if st.button("Start Transfer"):   
        directory = './outputs'
        bar = st.progress(0)
        run(update_progress_bar)
        concat = Image.open('./examples/style/concat_image.jpg')
        col2.image(concat)
        with st.container():
            st.image('./outputs/target_cat5_decoder_encoder_skip..jpg', use_column_width=True)
            st.session_state.process_idx = 4
    

if st.session_state.process_idx == 4:
    with open('./outputs/target_cat5_decoder_encoder_skip..jpg', 'rb') as file:
        button = st.download_button(label = 'Download', data = file, file_name = "Color_Grading.jpg", mime = 'image/jpg')


# 서버가 종료되지 않았다면, netstat -lnp | grep [포트번호] 후, kill -9 [process_id]

