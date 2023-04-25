import streamlit as st
from PIL import Image
from instagrapi import Client
from pathlib import Path
import requests
import os
import subprocess

def insta_crawling(ID, PW):
    cl = Client()
    cl.login(ID, PW)

    user_id = cl.user_id_from_username("jaeu8021")
    state_text.text("Feed searching...")

    medias = cl.user_medias(int(user_id), 9)

    folder = "test-folder"
    createDirectory(folder)
    state_text.text("Saving Image....")
    for m in medias:
        try:
            p = photo_download(cl, m.pk, folder)
        except AssertionError:
            pass
    
    state_text.text("Crawling finished! " + os.path.abspath(p))
    st.image(Image.open(os.path.abspath(p)))


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


def concat_image(files):  # test folder 에서 이미지를 받아와서 합해야됨

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

    def hconcat_resize_pil(im_list):
        msize = 1000
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
        msize = 1000
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

    concat_single_image = vconcat_pil(concat_row)
    st.image(concat_single_image)
    st.text(os.listdir('/examples'))
    concat_single_image.save('./examples/style/concat_image.jpg', 'JPEG')


st.title('AI color grader')
st.subheader('Find the filter that best fits your Instagram feed!')

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        uploaded_files = st.file_uploader(label="Choose image(s) for AI to analyze!",
                                          type=['jpeg', 'png', 'jpg', 'heic'],
                                          label_visibility='visible',
                                          accept_multiple_files=True)

    with col2:
        target_file = st.file_uploader(label="Choose an image to apply color correction!",
                                       type=['jpeg', 'png', 'jpg', 'heic'],
                                       label_visibility='visible',
                                       accept_multiple_files=False)
if target_file:
    target = Image.open(target_file)
    st.image(target)

crawled = []
# Check if the user has uploaded any files
if uploaded_files or crawled:
    # Create an empty list to store the images
    images = []

    # Loop through each uploaded file and append the opened image to the list
    for file in uploaded_files:
        image = Image.open(file)
        images.append(image)

    if st.button("Process Images!"):
        single = concat_image(images)
        st.write("Images are processed")

    if st.button("Start Analyzing!"):

        target.save(
            './examples/content/target.jpg', 'JPEG')
        st.write(type(target))


else:
    # If no files were uploaded, display a message
    st.write("Please upload one or more image files.")

# insta_id = st.text_input("Put your Instagram ID here!")
# insta_pwd = st.text_input('Put your Instagram password here!')
# # Instagram crawling button
# state_text = st.text("Ready to Crawl.")
# if st.button("Crawling Instagram"):
#     insta_crawling(insta_id, insta_pwd)

#id = "leessunj"
#pwd = "Ilsj08282!"

if st.button("Display the Output"):
    subprocess.run(['python3', 'transfer.py'])
    st.image('./outputs/target_cat5_decoder_encoder_skip..jpg')
