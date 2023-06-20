from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import tkinter as tk
from tkinter import ttk
import os
import glob
from PIL.ExifTags import TAGS
import torchvision.transforms.functional as TF
from model import UseModel
import torchvision.transforms as transforms
from DijkstraSkip import Dijkstra
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from aruco import Aruco
import csv
import torch
import numpy as np
from collections import deque
import time

# Maximum distance between two crack pixels
GAPWIDTH = 30

def get_creation_time(image_path):
    # Get the creation time from the EXIF data
    image = Image.open(image_path)
    exif_data = image._getexif()
    creation_time = None
    if exif_data:
        for tag, value in exif_data.items():
            if TAGS.get(tag) == 'DateTimeOriginal':
                creation_time = value
                break
    return creation_time


def run_model():
    # Run modules when "Run" button is pressed
    main_directory = foldername.get()
    # Get only JPG and PNG files in the directory
    image_files = glob.glob(os.path.join(main_directory, '*.jpg')) + glob.glob(os.path.join(main_directory, '*.png'))
    # Sort the files based on creation time
    try:
        sorted_files = sorted(image_files, key=lambda f: get_creation_time(f))
        sorted_files = sorted_files[::skip_no.get()+1]
    except TypeError:
        sorted_files = image_files
    for file_index,sorted_file in enumerate(sorted_files):
        root.update_idletasks()
        time.sleep(5)
        filepath.set(sorted_file)
        img = Image.open(sorted_file)
        img_w,img_h = img.size
        # Resizing image 
        if img_w > img_h:
            conv_h = int(img_h*2048/img_w)
            # Setting values such as image can be downsampled and upsampled by the model smoothly
            conv_h = int(conv_h /16)
            conv_h = conv_h *16
            img = img.resize((2048,conv_h))
        else:
            conv_w = int(img_w*2048/img_h)
            conv_w = int(conv_w /16)
            conv_w = conv_w *16
            img = img.resize((conv_w,2048))
        img_tf = TF.to_tensor(img)
        new_image = transforms.ToPILImage()(img_tf)
        new_filename = sorted_file.split("\\")[-1]
        new_filepath = os.path.join(outputfolder.get(), new_filename)
        # Saving new resized image
        new_image.save(new_filepath, "PNG")
        # Performing prediction
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        unet.model.to(device)
        img_tf = img_tf.to(device)
        out = unet.predict(img_tf)
        #print("Prediction done")
        out = out.type(torch.uint8)
        seg_image = transforms.ToPILImage()(out)
        seg_filename = sorted_file.split("\\")[-1]
        last_dot_index = seg_filename.rfind(".")
        seg_filename = seg_filename[:last_dot_index]
        file_name_list.append(seg_filename)
        seg_filename_ext = seg_filename + "_segment.png"
        seg_filepath = os.path.join(outputfolder.get(), seg_filename_ext)
        # Saving segmentation image
        seg_image.save(seg_filepath, "PNG")
        # Performing post processing
        process_image = filter(seg_image,GAPWIDTH)
        process_image_filename_ext = seg_filename + "_segment_post_process.png"
        process_image_filepath = os.path.join(outputfolder.get(), process_image_filename_ext)
        # Saving post processed segmentation image
        process_image.save(process_image_filepath, "PNG")
        # Calculating the crack length
        crack_len,crack_path_img,start_pixel,end_pixel = dij.predict(seg_filepath)
        # Performing Aruco detection
        aruco_return = aruco.predict(new_filepath,sqrlen.get(),marklen.get())
        aruco_location = (None)
        #print("Aruco calculation done")
        if aruco_return is not None:
            aruco_value, aruco_diamond_location = aruco_return
            pixelmul.set(aruco_value)
            aruco_location = (aruco_diamond_location)
        aruco_location_list.append(aruco_location)
        # Performing pixel-to-mm conversion
        crack_len_max = max(crack_len) * pixelmul.get()
        crack_path_img = Image.fromarray(crack_path_img)
        crack_path_filename_ext = seg_filename + "_path.png"
        crack_path_filepath = os.path.join(outputfolder.get(), crack_path_filename_ext)
        # saving the crack path highlighted image
        img.save(crack_path_filepath,"PNG")
        outfilepath.set(crack_path_filepath)
        cracklength.set(crack_len_max)
        cracklength_list_max.append(crack_len_max)
        cracklength_list[file_index] = crack_len
        start_pixel_list[file_index] = start_pixel
        end_pixel_list[file_index] = end_pixel
        #print("Crack length found")
        # Calculating Crack Propagation curve
        create_line_graph()
    save_button["state"] = "normal"
    #print("Done the process")

def display_img(*args):
    # Display input image in the placeholder in the gui
    if filepath.get() != '':
        image = Image.open(filepath.get())
        image = image.resize((400, 300))
        tk_image = ImageTk.PhotoImage(image)
        image_label.config(image=tk_image)
        image_label.image = tk_image
    else:
        # Display grey image if there is no image to display
        image = Image.open("images/greybg.png")
        image = image.resize((400, 300))
        tk_image = ImageTk.PhotoImage(image)
        image_label.config(image=tk_image)
        image_label.image = tk_image

def display_img_out(*args):
    # Display input image in the placeholder in the gui
    if outfilepath.get() != '':
        image = Image.open(outfilepath.get())
        image = image.resize((400, 300))
        tk_image = ImageTk.PhotoImage(image)
        out_image_label.config(image=tk_image)
        out_image_label.image = tk_image
    else:
        # Display grey image if there is no image to display
        image = Image.open("images/greybg.png")
        image = image.resize((400, 300))
        tk_image = ImageTk.PhotoImage(image)
        out_image_label.config(image=tk_image)
        out_image_label.image = tk_image

def save_foldername():
    # Update the folder location when "Select Folder" is clicked
    filename = filedialog.askdirectory(initialdir="/", title="Select Folder")
    if filename:
        foldername.set(filename)
        if cyclenum.get() != 0:
            run_button["state"] = "normal"
        output_path = os.path.join(filename, "Output")
        outputfolder.set(output_path)
        try:
            os.makedirs(output_path)
            #print("New directory created successfully!")
        except OSError as e:
            print(f"Failed to create directory: {e}")
        print(foldername.get())

def create_line_graph(tosave:bool = False):
    # Create the crack propagation curve in the gui
    fig = Figure(figsize=(6, 4), dpi=100)
    ax = fig.add_subplot(111)
    if len(cracklength_list) == 0:
        x = [0]
        y = [0]
    else:
        x = [cyclenum.get() * i for i in range(1, len(cracklength_list)+1)] 
        y = cracklength_list_max
    ax.plot(x, y, '-o')
    ax.set_xlabel('Number of Cycles')
    ax.set_ylabel('Crack Length mm')

    if tosave:
        # Generate a csv file with calculated results
        plot_path = os.path.join(outputfolder.get(), "Plot.png")
        fig.savefig(plot_path)
        csv_filepath = os.path.join(outputfolder.get(), 'data.csv')
        with open(csv_filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['No of cycles', 'Crack length in mm','Start pixel','End pixel','Image Name','Aruco Codinates'])
            for key, value2 in cracklength_list.items():
                value1 = x[key]
                value3 = start_pixel_list[key]
                value4 = end_pixel_list[key]
                value5 = file_name_list[key]
                value6 = aruco_location_list[key]
                writer.writerow([value1, value2, value3, value4, value5, value6])
        print(f"CSV file '{csv_filepath}' has been created.")
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().place(x=950, y=70)

def get_cycle_num():
    # Acquire the number of cycles from the gui
    number = cycle_num_entry.get()
    try:
        number = int(number)
        cyclenum.set(number)
        if foldername.get() != "" and number != 0:
            run_button["state"] = "normal"
        elif number == 0:
            run_button["state"] = "disabled"
    except ValueError:
        print("Value Error: Enter integer numbers")

def get_skip_num():
    # Acquire the number of cycles from the gui
    number = skip_num_entry.get()
    try:
        number = int(number)
        skip_no.set(number)
    except ValueError:
        print("Value Error: Enter integer numbers")

def get_pixel_rate():
    # Acquire the pixel-to-mm ratio from the gui
    number = aruco_entry.get()
    try:
        number = float(number)
        pixelmul.set(number)
    except ValueError:
        print("Value Error: Enter numbers only")

def get_squarelength():
    # Acquire the squarelength parameter from the gui
    number = sqrlen_entry.get()
    try:
        number = float(number)
        sqrlen.set(number)
    except ValueError:
        print("Value Error: Enter numbers only")

def get_markerlength():
    # Acquire the markerlength parameter from the gui
    number = marklen_entry.get()
    try:
        number = float(number)
        marklen.set(number)
    except ValueError:
        print("Value Error: Enter numbers only")

def update_ar(*args):
    # Switch the button enabled and disabled based on the checkmark
    if checkbox_var.get():
        aruco_button["state"] = "disabled"
        aruco_entry["state"] = "disabled"
        sqrlen_button["state"] = "normal"
        sqrlen_entry["state"] = "normal"
        marklen_button["state"] = "normal"
        marklen_entry["state"] = "normal"
        
    else:
        aruco_button["state"] = "normal"
        aruco_entry["state"] = "normal"
        sqrlen_button["state"] = "disabled"
        sqrlen_entry["state"] = "disabled"
        marklen_button["state"] = "disabled"
        marklen_entry["state"] = "disabled"

def filter(img: Image, threshold):
    # Perform the post processing on the predicted image
    img = image_to_2d_list(img)
    length, width = len(img), len(img[0])
    vis = [[0 for i in range(width)] for j in range(length)]
    ans = []
    out = [[0 for i in range(width)] for j in range(length)]
    # Assign different values for distanct cracks in the predicted image
    for i in range(length):
        for j in range(width):
            if img[i][j] == 255 and not vis[i][j]:
                tmp = []
                def bfs(i, j):
                    q = deque()
                    q.append((i, j))
                    vis[i][j] = 1
                    tmp.append((i, j))
                    while len(q):
                        top_i, top_j = q.pop()
                        for i in range(max(0, top_i - threshold), min(length, top_i + threshold)):
                            for j in range(max(0, top_j - threshold), min(width, top_j + threshold)):
                                if not vis[i][j] and img[i][j] == 255:
                                    q.append((i, j))
                                    vis[i][j] = 1
                                    tmp.append((i, j))
                bfs(i, j)
                ans.append(tmp)
    for id in range(len(ans)):
        for i, j in ans[id]:
            out[i][j] = id + 1
    out = list_to_gray_image(out)
    return out


def list_to_gray_image(data):
    # Convert image to greyscale
    height = len(data)
    width = len(data[0])
    image = Image.new("L", (width, height))
    pixels = []
    for row in data:
        for value in row:
            pixels.append(value)
    image.putdata(pixels)
    return image


def image_to_2d_list(image):
    # Create a 2D list representation of the image
    pixel_data = list(image.getdata())
    width, height = image.size
    pixels_2d = [pixel_data[i:i+width] for i in range(0, len(pixel_data), width)]
    return pixels_2d

root = tk.Tk()
root.geometry('1500x700')
root.title('Segment Crack')

#Initializing sub-modules

unet = UseModel("weights/best_epoch_weights (4).pth")
dij = Dijkstra(gapwidth=GAPWIDTH)
aruco = Aruco()

# Variable to save folder path
foldername = tk.StringVar()
# Variable to save crack length
cracklength = tk.StringVar()
# Variable to save cycle number
cyclenum = tk.IntVar()
# Variable to save pixel conversion rate
pixelmul = tk.DoubleVar()
pixelmul.set(1)
# Variable to save squarelength and markerlength
sqrlen = tk.DoubleVar()
sqrlen.set(0.00341)
marklen = tk.DoubleVar()
marklen.set(0.0025)

#List of crack length
cracklength_list = {}
cracklength_list_max = []
start_pixel_list = {}
end_pixel_list = {}
file_name_list = []
aruco_location_list = []

# Variable to save output folder path
outputfolder = tk.StringVar()
# Variable to save skip number
skip_no = tk.IntVar()
skip_no.set(0)
# Variable to save input current file path
filepath = tk.StringVar()
filepath.trace("w", display_img)
# Variable to save output current file path
outfilepath = tk.StringVar()
outfilepath.trace("w", display_img_out)

# Button to select folder path
folder_distination_button = ttk.Button(
    root,
    text='Select Folder',
    command=lambda: save_foldername()
)
folder_distination_button.place(x=10, y=10)

# Displaying Folder Path
display_folder_name = tk.Entry(root,textvariable=foldername, state="readonly",width=50)
display_folder_name.insert(0, foldername)
display_folder_name.place(x=10,y=40)

# Display input image
image_label = tk.Label(root, relief="solid", borderwidth=1, bg="gray")
filepath.set("")
image_label.place(x=10,y=70)

# Display output image
out_image_label = tk.Label(root, relief="solid", borderwidth=1, bg="gray")
outfilepath.set("")
out_image_label.place(x=500,y=70)

# Run Button
run_button = ttk.Button(
    root,
    text='Run',
    command=lambda: run_model()
)
run_button.place(x=10, y=400)
run_button["state"] = "disabled"

#checkbox for aruco
checkbox_var = tk.BooleanVar()
checkbox_var.trace('w', update_ar)
checkbox = tk.Checkbutton(root, text="Aruco present", variable=checkbox_var)
checkbox.place(x=10, y=430)
 
# Displaying Crack Length
crack_length_label = tk.Label(root, text="Crack Length")
crack_length_label.place(x=500,y=10)
display_crack_length = tk.Entry(root,textvariable=cracklength, state="readonly",width=50)
display_crack_length.insert(0, cracklength)
display_crack_length.place(x=500,y=40)

# Displaying Crack Length Graph
crack_graph_label = tk.Label(root, text="Crack Propagation Graph")
crack_graph_label.place(x=950,y=10)

# Label and Entry for number of cycles
cycle_num_label = tk.Label(root, text="Enter the cycle number:")
cycle_num_label.place(x=500, y=400)
cycle_num_entry = tk.Entry(root,textvariable=cyclenum)
cycle_num_entry.place(x=500, y=430)
cycle_num_button = tk.Button(root, text="Enter", command=get_cycle_num)
cycle_num_button.place(x=650, y=425)

# Label and Entry for number of files to be skipped
skip_num_label = tk.Label(root, text="Enter the skip number:")
skip_num_label.place(x=150, y=400)
skip_num_entry = tk.Entry(root,textvariable=skip_no)
skip_num_entry.place(x=150, y=430)
skip_num_button = tk.Button(root, text="Enter", command=get_skip_num)
skip_num_button.place(x=300, y=425)

# Label and Entry for pixel conversion rate
aruco_label = tk.Label(root, text="Enter the pixel conversion rate if aruco label is not present:")
aruco_label.place(x=500, y=460)
aruco_entry = tk.Entry(root,textvariable=pixelmul)
aruco_entry.place(x=500, y=490)
aruco_button = tk.Button(root, text="Enter", command=get_pixel_rate)
aruco_button.place(x=650, y=485)

# Label and Entry for aruco squarelength and markerlength
sqrlen_label = tk.Label(root, text="Enter the aruco squarelength")
sqrlen_label.place(x=500, y=520)
sqrlen_entry = tk.Entry(root,textvariable=sqrlen)
sqrlen_entry.place(x=500, y=550)
sqrlen_button = tk.Button(root, text="Enter", command=get_squarelength)
sqrlen_button.place(x=650, y=545)
marklen_label = tk.Label(root, text="Enter the aruco markerlength")
marklen_label.place(x=500, y=580)
marklen_entry = tk.Entry(root,textvariable=marklen)
marklen_entry.place(x=500, y=610)
marklen_button = tk.Button(root, text="Enter", command=get_markerlength)
marklen_button.place(x=650, y=605)
checkbox_var.set(False)

create_line_graph()

# Save Button
save_button = ttk.Button(
    root,
    text='Save',
    command=lambda: create_line_graph(tosave=True)
)
save_button.place(x=950, y=485)
save_button["state"] = "disabled"

root.mainloop()