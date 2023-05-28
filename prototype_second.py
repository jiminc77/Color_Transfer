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
import time

from color_matcher import ColorMatcher
from color_matcher.io_handler import load_img_file, save_img_file, FILE_EXTS
from color_matcher.normalizer import Normalizer

def color_matcher(src,ref):
    img_src=load_img_file(src)
    img_ref=load_img_file(ref)
    cm = ColorMatcher()
    img_res = cm.transfer(src=img_src, ref=img_ref, method='mkl')
    img_res = Normalizer(img_res).uint8_norm()
    save_img_file(img_res, os.path.join('outputs', f'{st.session_state.seed}_colormatch.png'))
    return img_res


### streamlit style options

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

###utils

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

def display_available_memory():
    mem_info = psutil.virtual_memory()
    available_memory = mem_info.available / (1024 ** 2)  # Convert to MB
    st.write(f"Available memory: {available_memory:.2f} MB")

def insta_crawling(ID, PW,target="jaeu8021"):
    cl = Client()
    cl.login(ID, PW)
    crawl_state.text("Log in...")
    user_id = cl.user_id_from_username(target)
    crawl_state.text("Feed searching...")

    medias = cl.user_medias_v1(int(user_id), 9)
    print(len(medias))
    if len(medias)<1:
        crawl_state.markdown(f"There're **No** photos: {target}")
        return
    folder = f"{st.session_state.seed}_test-folder"
    createDirectory(folder)
    
    temp = []
    crawl_state.text(f"Saving Image....({len(temp)})")
    for m in medias:
        try:
            p = photo_download(cl, m.pk, folder)
            temp.append(p)
        except AssertionError:
            pass
        crawl_state.text(f"Saving Image....({len(temp)})")
    crawl_state.text("Crawling finished! ") # + os.path.abspath(p))
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
    if n<1:
        return "NO-images"
    for i in range(0, n, 3):
        if n-i < 3:
            break
        row = hconcat_resize_pil(images[i:i+3],msize)
        concat_row.append(row)
        # progress = (i + 3) / n
        # progress_callback(progress)
    if not concat_row:
        concat_single_image=images[0]
    else:
        concat_single_image = vconcat_pil(concat_row,msize)
    # st.image(concat_single_image)
    createDirectory('examples/style')
    createDirectory('examples/style_segment')
    createDirectory('examples/content')
    createDirectory('examples/content_segment')
    shutil.copyfile('black_.png', 'examples/style_segment/black_.png')
    shutil.copyfile('black_.png', 'examples/content_segment/black_.png')

    concat_single_image.save(f'./examples/style/{st.session_state.seed}_concat_image.jpg', 'JPEG')
    return "concat-saved"

def update_progress_bar(progress):
    
    if progress < 0.99:
        bar.progress(progress)
    else:
        bar.progress(progress)
        time.sleep(1)
        bar.empty()
        

def delete_folder(filepath):
    if os.path.exists(filepath):
        shutil.rmtree(filepath)
        # os.rmdir(filepath)
        print("delete")
        return "Remove folder"
    else:
        return "Directory Not Found"

def delete_files(filelist):
    for file in filelist:
        if os.path.exists(file):
            os.remove(file)
    return "Remove All File"

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

def get_images(li):
    imgs=[]
    for file in li:
        image = Image.open(file)
        imgs.append(image)
    return imgs

def concating():
    images=st.session_state.images
    print("concat-processing!!!")
    # bar = st.progress(0)
    single = concat_image(images, update_progress_bar)
    st.session_state.process_idx = 3

def toggle_imethod():
    st.session_state.imethod=0 if st.session_state.imethod else 1

### streamlit vars

if 'process_idx' not in st.session_state:
    st.session_state.process_idx = 1
if 'crawled' not in st.session_state:
    st.session_state.crawled=[]
if 'uploaded' not in st.session_state:
    st.session_state.uploaded=[]
if 'images' not in st.session_state:
    st.session_state.images=[]
if 'target' not in st.session_state:
    st.session_state.target=None
if 'ref' not in st.session_state:
    st.session_state.ref=0
if 'seed' not in st.session_state:
    st.session_state.seed=0
if 'imethod' not in st.session_state:
    st.session_state.imethod=0 #default crawling(0), uploading(1)

### stramlit UI

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
            if not st.session_state.seed:
                st.session_state.seed=time.time()
                print(st.session_state.seed)
        
            if not is_square(target_image):
                st.error("Please upload a square image.")
            else:
                pass
    
    with col2:
        print(st.session_state.seed)
        option=st.selectbox("Get images for AI to anaylze",
                            ("By Instagram crawling","By Uploading"),
                            on_change=toggle_imethod, index=st.session_state.imethod)
        
        if st.session_state.imethod==0: #crawling
            st.text("Get images for AI to anaylze by Instagram login")
            with st.form("crawling"):
                insta_id = st.text_input("Put your Instagram ID here!")
                insta_pwd = st.text_input('Put your Instagram password here!',type='password')
            
                username = st.text_input("Put target Instagram ID here if you want!",placeholder="default:your_id")
                
                submitted = st.form_submit_button("Submit")
                if submitted:
                    if not username:
                        username=insta_id
                    st.write("Crawling photos from ",username)
                    crawl_state=st.text("...")
                    insta_crawling(insta_id, insta_pwd,target=username)
                    concating()

        elif st.session_state.imethod==1:
            st.session_state.uploaded = st.file_uploader(label="Choose image(s) for AI to analyze",
                                          type=['jpeg', 'png', 'jpg', 'heic'],
                                          label_visibility='visible',
                                          accept_multiple_files=True)
            if st.button("Process Images", type="primary"):
                concating()
        

with st.container():
    ic1, ic2 = st.columns(2)
    print(st.session_state.process_idx)
    if target_file:
        target = Image.open(target_file)
        # here!
        st.write(os.listdir('/app/color_transfer/examples/content'))
        target.save(f'/app/color_transfer/examples/content/{st.session_state.seed}_target.jpeg', 'JPEG')
        # target.save(f"/examples/content/target.jpg", 'JPEG')
        with ic1:
            # st.markdown('<div class="custom-style"></div>', unsafe_allow_html=True)
            st.markdown("**target image**")
            st.image(target)
    with ic2:
        ref_state=st.markdown("")
        if st.session_state.process_idx == 3:
            if not os.path.exists(f'examples/style/{st.session_state.seed}_concat_image.jpg'):
                st.session_state.process_idx=1
                ref_state.markdown("**Error**: try again getting reference images")
            else:   
                ref=Image.open(f'examples/style/{st.session_state.seed}_concat_image.jpg')
                st.image(ref)


if st.session_state.crawled:
    st.session_state.images=get_images(st.session_state.crawled)
    ref_state.markdown("**reference images from CRAWLING**")
if st.session_state.uploaded:
    st.session_state.images=get_images(st.session_state.uploaded)
    ref_state.markdown("**reference images from uploading**")
    if st.session_state.process_idx<2:
        st.session_state.process_idx = 2
    
        
st.write(st.session_state.process_idx)

if st.session_state.process_idx == 3 :#and target_file and st.session_state.images
    if st.button("Start Transfer", type="primary",disabled= not target_file or not st.session_state.images,help="shoud need target image and ref images"):   
        directory = 'outputs'
        # color_matcher(f'./examples/content/{st.session_state.seed}_target.jpg',f'./examples/style/{st.session_state.seed}_concat_image.jpg')
        # st.image(f'./outputs/{st.session_state.seed}_colormatch.png')
        bar = st.progress(0)
        run(update_progress_bar,seed=st.session_state.seed)
        with st.container():
            st.image(f'outputs/{st.session_state.seed}_target_cat5_decoder_encoder_skip..jpg', use_column_width=True)
            st.session_state.process_idx = 4
    

if st.session_state.process_idx == 4:
    with open(f'outputs/{st.session_state.seed}_target_cat5_decoder_encoder_skip..jpg', 'rb') as file:
        button = st.download_button(label = 'Download', data = file, file_name = "Color_Grading.jpg", mime = 'image/jpg')


if st.button("finish"):
    st.session_state.process_idx = 1
    print(st.session_state.seed)
    delete_files([f'examples/style/{st.session_state.seed}_concat_image.jpg',f'examples/content/{st.session_state.seed}_target.jpg',f'outputs/{st.session_state.seed}_target_cat5_decoder_encoder_skip..jpg'])
    # delete_folder(f"{st.session_state.seed}_test-folder")
    st.experimental_rerun()


# 서버가 종료되지 않았다면, netstat -lnp | grep [포트번호] 후, kill -9 [process_id]

