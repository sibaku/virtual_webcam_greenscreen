# based on the webcam demo provided by https://github.com/ZHKKKe/MODNet

from enum import Enum
import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import threading
import time
from src.models.modnet import MODNet
import pyvirtualcam
from tkinter import filedialog as fd
from tkinter import colorchooser

# the following code is not super clean but hopefully co
root = tk.Tk()
canvas = tk.Canvas(root, bg="#fff", height=360, width=640)
canvas.pack()

canvas_lock = lock = threading.Lock()

# frame data used to convert to preview image
rgb = np.uint8(np.full((360, 640, 3), 255.0))
# store data globally, since it will be deleted otherwise, as tkinter does not take ownership...
PIL_image = Image.fromarray(np.uint8(rgb)).convert('RGB')
tk_image = ImageTk.PhotoImage(PIL_image)
# image container to draw preview
image_container = canvas.create_image(0, 0, anchor="nw", image=tk_image)

# helper function


def clamp(x, a, b):
    return max(a, min(b, x))

# provides solid color background


class BGColor:
    def __init__(self, w, h, r=255, g=255, b=255):
        # cache data
        self.color = (clamp(r, 0, 255), clamp(g, 0, 255), clamp(b, 0, 255))
        self.data = np.zeros((h, w, 3), np.uint8)
        self.data[:] = self.color

    def get(self, frame):
        # resize if needed, otherwise return cached solid color background
        h, w, _ = frame.shape
        if w != self.data.shape[1] or h != self.data.shape[0]:
            self.data = np.zeros((h, w, 3), np.uint8)
            self.data[:] = self.color
        return self.data

# provides a background image


class BGImage:
    def __init__(self, filename):
        # load image and cache resized image
        self.img = cv2.imread(filename, cv2.IMREAD_COLOR)
        if self.img is None:
            raise Exception(f'Could not load file: {filename}')
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        self.draw_img = self.img

    def get(self, frame):
        # resize if needed, otherwise return cached image
        if frame.shape[0] != self.draw_img.shape[0] or frame.shape[1] != self.draw_img.shape[1]:
            self.draw_img = cv2.resize(
                self.img, (frame.shape[1], frame.shape[0]), cv2.INTER_AREA)
        return self.draw_img

# blurs the background


class BGBlur:
    def __init__(self, blur_amount=(21, 21)):
        # blur filter size
        self.blur_amount = blur_amount

    def get(self, frame):
        # resize and blur to both reduce computation and get a nice strong blur
        frame_small = cv2.resize(
            frame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        frame_small = cv2.blur(frame_small, self.blur_amount)
        frame_small = cv2.GaussianBlur(frame_small, self.blur_amount, 0)
        return cv2.resize(
            frame_small, (frame.shape[1], frame.shape[0]), cv2.INTER_AREA)

# provides a video as background


class BGVideo:
    def __init__(self, filename):
        # load video and get fps
        self.cap = cv2.VideoCapture(filename)
        if not self.cap.isOpened():
            raise Exception(f'Could not load file: {filename}')
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.current_frame = 0
        self.start_time = -1

    def get(self, frame):
        # update run time
        if self.start_time < 0:
            self.start_time = time.time()
        cur_time = time.time()
        delta = cur_time - self.start_time
        self.start_time = cur_time
        self.current_frame += self.fps * delta
        # update video time
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.current_frame))

        # read video frame
        ret, f = self.cap.read()
        if ret != True:
            # video has finished -> try to reset
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame = 0
            ret, f = self.cap.read()
        # video still not working -> go with a white background
        if ret == False:
            f = np.full(frame.shape, 255.0)

        # convert color
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        f = cv2.resize(
            f, (frame.shape[1], frame.shape[0]), cv2.INTER_AREA)
        return f


bg_lock = lock = threading.Lock()

bg = BGColor(1280, 720, 0, 255, 0)


# enable/disable background
remove_bg = True
bg_toggle_var = tk.IntVar()


def callback_toggle():
    global remove_bg
    with bg_lock:
        remove_bg = bool(bg_toggle_var.get())


bg_frame_options = tk.Frame(root)
bg_frame_options.pack(expand=True, fill=tk.BOTH, anchor='center')

toggle_button = tk.Checkbutton(
    bg_frame_options, text="Remove BG", variable=bg_toggle_var, onvalue=1, offvalue=0, command=callback_toggle)
toggle_button.pack(side=tk.LEFT)
toggle_button.select()


class CropType(Enum):
    CENTER = 0
    FULL = 1


crop_type = CropType.CENTER


def callback_crop():
    global crop_type
    with bg_lock:
        toggle_crop_button.config(text=f'Set crop type to: {crop_type.name}')
        crop_type = CropType.FULL if crop_type == CropType.CENTER else CropType.CENTER


toggle_crop_button = tk.Button(
    bg_frame_options,
    text=f'Set crop type to: {CropType.FULL.name}',
    command=callback_crop
)
toggle_crop_button.pack(side=tk.LEFT)


disable_cam_input = False
fg_toggle_var = tk.IntVar()


def callback_fg_toggle():
    global disable_cam_input
    with bg_lock:
        disable_cam_input = bool(fg_toggle_var.get())


toggle_disbale_fg_button = tk.Checkbutton(
    bg_frame_options, text="Disable cam input", variable=fg_toggle_var, onvalue=1, offvalue=0, command=callback_fg_toggle)
toggle_disbale_fg_button.pack(side=tk.LEFT)
toggle_disbale_fg_button.deselect()


def callback_img():
    global bg

    filename = fd.askopenfilename()
    if not filename:
        return
    # do something
    try:
        with bg_lock:
            bg_temp = BGImage(filename)

            bg = bg_temp

    except:
        print(f'Could not open file {filename}')


bg_frame_widget = tk.Frame(root)
bg_frame_widget.pack(expand=True, fill=tk.BOTH, anchor='center')


def callback_video():
    global bg

    filename = fd.askopenfilename()
    if not filename:
        return
    # do something
    try:
        with bg_lock:
            bg_temp = BGVideo(filename)
            bg = bg_temp
    except:
        print(f'Could not open file {filename}')


def callback_color():
    global bg

    color_code = colorchooser.askcolor(title="Choose color")
    if color_code is None:
        return
    col = color_code[0]
    if col is None:
        return
    with bg_lock:
        bg = BGColor(1280, 720, col[0], col[1], col[2])


def callback_blur():
    global bg

    # do something
    with bg_lock:
        bg = BGBlur()


img_button = tk.Button(
    bg_frame_widget,
    text="Image",
    command=callback_img
)
img_button.pack(side=tk.LEFT)
vid_button = tk.Button(
    bg_frame_widget,
    text="Video",
    command=callback_video
)
vid_button.pack(side=tk.LEFT)

col_button = tk.Button(
    bg_frame_widget,
    text="Color",
    command=callback_color
)
col_button.pack(side=tk.LEFT)

blur_button = tk.Button(
    bg_frame_widget,
    text="Blur",
    command=callback_blur
)
blur_button.pack(side=tk.LEFT)


# runs the actual extraction code

run_cam_thread = True


def cam_thread():
    torch_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    print('Load pre-trained MODNet...')
    pretrained_ckpt = './pretrained/modnet_webcam_portrait_matting.ckpt'
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)

    GPU = True if torch.cuda.device_count() > 0 else False
    if GPU:
        print('Use GPU...')
        modnet = modnet.cuda()
        modnet.load_state_dict(torch.load(pretrained_ckpt))
    else:
        print('Use CPU...')
        modnet.load_state_dict(torch.load(
            pretrained_ckpt, map_location=torch.device('cpu')))

    modnet.eval()

    print('Init WebCam...')
    cap = cv2.VideoCapture(0)
    print('Opened Webcam')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print('Start matting...')
    # open virtual camera
    with pyvirtualcam.Camera(width=1280, height=720, fps=20) as cam:
        global rgb
        print(f'Using virtual camera: {cam.device}')

        while(run_cam_thread):

            # save work when only background is shown
            if disable_cam_input:
                bg_maker = None
                with bg_lock:
                    bg_maker = bg

                # only background
                frame_np = np.uint8(np.full((720, 1280, 3), 0.0))
                frame_np = bg_maker.get(frame_np)
                with canvas_lock:
                    rgb = frame_np
                cam.send(frame_np)
                cam.sleep_until_next_frame()
                continue

            _, frame_np = cap.read()

            frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
            frame_np = cv2.flip(frame_np, 1)
            frame_origin = frame_np
            with bg_lock:
                current_crop = crop_type

            # disabled bg removal -> just return the camera
            if not remove_bg:
                # account for either cropped or full mode
                if current_crop == CropType.CENTER:
                    # cropped region so that matting result and camera image coincide
                    frame_np = frame_np[94:94+531, 167:167+945, :]
                    frame_np = cv2.resize(
                        frame_np, (1280, 720), cv2.INTER_AREA)

                with canvas_lock:
                    rgb = frame_np
                cam.send(frame_np)
                cam.sleep_until_next_frame()
                continue

            if current_crop == CropType.CENTER:
                # network input has matrix input size 512x672
                # cropped mode tries to fill out the whole input with the camera,
                # which seems to provide better results, but throws away outside of the region
                frame_resized = cv2.resize(
                    frame_np, (910, 512), cv2.INTER_AREA)
                frame_np = np.uint8(np.full((512, 672, 3), 0.0))
                frame_np[:, :, :] = frame_resized[:, 119:119+672, :]
            else:
                # put all data resized into the network input
                # to account for aspect ratio, there will be some empty regions,
                # which seem to slightly worsen the result
                frame_resized = cv2.resize(
                    frame_np, (672, 378), cv2.INTER_AREA)
                frame_np = np.uint8(np.full((512, 672, 3), 0.0))
                frame_np[67:378+67, :, :] = frame_resized

            # apply network
            frame_PIL = Image.fromarray(frame_np)
            frame_tensor = torch_transforms(frame_PIL)
            frame_tensor = frame_tensor[None, :, :, :]
            if GPU:
                frame_tensor = frame_tensor.cuda()

            with torch.no_grad():
                _, _, matte_tensor = modnet(frame_tensor, True)

            matte_tensor = matte_tensor.repeat(1, 3, 1, 1)
            matte_np = matte_tensor[0].data.cpu().numpy().transpose(1, 2, 0)

            if current_crop == CropType.CENTER:
                # extract subregion with the correct camera aspect ratio
                matte_np = matte_np[67:378+67, :, :]
                matte_np = cv2.resize(matte_np, (1280, 720), cv2.INTER_AREA)
                # crop out the processed region
                frame_resized = frame_origin[94:94+531, 167:167+945, :]
                frame_resized = cv2.resize(
                    frame_resized, (1280, 720), cv2.INTER_AREA)
            else:
                # extract the actual data from the result
                matte_np = matte_np[67:378+67, :, :]
                matte_np = cv2.resize(matte_np, (1280, 720), cv2.INTER_AREA)
                frame_resized = frame_origin

            # get bg provider
            bg_maker = None
            with bg_lock:
                bg_maker = bg

            # apply mask
            bg_img = bg_maker.get(frame_resized)
            fg_np = matte_np * frame_resized + \
                (1 - matte_np) * bg_img

            # cam_frame = cv2.resize(fg_np, (1280, 720), cv2.INTER_AREA)
            cam_frame = np.uint8(fg_np)

            # set frame for preview
            with canvas_lock:
                rgb = cam_frame

            # send data to camera
            cam.send(cam_frame)
            cam.sleep_until_next_frame()
        cap.release()


# start camera thread
ct = threading.Thread(target=cam_thread, daemon=True)
ct.start()

# update gui preview image


def update_gui():
    im = None
    with canvas_lock:
        im = rgb
    if (im is not None):
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        scale_factor = min(w / im.shape[1], h/im.shape[0])
        new_height = int(im.shape[0] * scale_factor)
        new_width = int(im.shape[1] * scale_factor)
        dimensions = (new_width, new_height)
        im = cv2.resize(im, dimensions, cv2.INTER_AREA)
        im2 = Image.fromarray(np.uint8(im)).convert('RGB')
        im_tk = ImageTk.PhotoImage(im2)
        global tk_image
        tk_image = im_tk
        canvas.itemconfig(image_container, image=im_tk)
        canvas.pack()
    root.after(200, update_gui)


root.after(1000, update_gui)
root.title("Virtual greenscreen")
root.mainloop()

run_cam_thread = False
ct.join(5)
print('Exit...')
