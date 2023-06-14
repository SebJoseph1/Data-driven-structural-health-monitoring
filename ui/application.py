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
    main_directory = foldername.get()
    # Get only JPG and PNG files in the directory
    image_files = glob.glob(os.path.join(main_directory, '*.jpg')) + glob.glob(os.path.join(main_directory, '*.png'))
    # Sort the files based on creation time
    try:
        sorted_files = sorted(image_files, key=lambda f: get_creation_time(f))
    except TypeError:
        sorted_files = image_files
    for sorted_file in sorted_files:
        root.update_idletasks()
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
        crack_len,crack_path_img,start_pixel,end_pixel = dij.predict(seg_filepath)
        aruco_return = aruco.predict(new_filepath,sqrlen.get(),marklen.get())
        aruco_location = (None)
        #print("Aruco calculation done")
        if aruco_return is not None:
            aruco_value, aruco_diamond_location = aruco_return
            pixelmul.set(aruco_value)
            aruco_location = (aruco_diamond_location)
        aruco_location_list.append(aruco_location)
        crack_len = crack_len * pixelmul.get()
        crack_path_img = Image.fromarray(crack_path_img)
        crack_path_filename_ext = seg_filename + "_path.png"
        crack_path_filepath = os.path.join(outputfolder.get(), crack_path_filename_ext)
        img.save(crack_path_filepath,"PNG")
        outfilepath.set(crack_path_filepath)
        cracklength.set(crack_len)
        cracklength_list.append(crack_len)
        start_pixel_list.append(start_pixel)
        end_pixel_list.append(end_pixel)
        #print("Crack length found")
        create_line_graph()
    save_button["state"] = "normal"
    #print("Done the process")


        

def display_img(*args):
    if filepath.get() != '':
        image = Image.open(filepath.get())
        image = image.resize((400, 300))
        tk_image = ImageTk.PhotoImage(image)
        image_label.config(image=tk_image)
        image_label.image = tk_image
    else:
        image = Image.open("images/greybg.png")
        image = image.resize((400, 300))
        tk_image = ImageTk.PhotoImage(image)
        image_label.config(image=tk_image)
        image_label.image = tk_image

def display_img_out(*args):
    if outfilepath.get() != '':
        image = Image.open(outfilepath.get())
        image = image.resize((400, 300))
        tk_image = ImageTk.PhotoImage(image)
        out_image_label.config(image=tk_image)
        out_image_label.image = tk_image
    else:
        image = Image.open("images/greybg.png")
        image = image.resize((400, 300))
        tk_image = ImageTk.PhotoImage(image)
        out_image_label.config(image=tk_image)
        out_image_label.image = tk_image

def save_foldername():
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
    fig = Figure(figsize=(6, 4), dpi=100)
    ax = fig.add_subplot(111)
    x = [cyclenum.get() * i for i in range(1, len(cracklength_list)+1)] 
    y = cracklength_list
    ax.plot(x, y, '-o')
    ax.set_xlabel('Number of Cycles')
    ax.set_ylabel('Crack Length mm')

    if tosave:
        plot_path = os.path.join(outputfolder.get(), "Plot.png")
        fig.savefig(plot_path)
        csv_filepath = os.path.join(outputfolder.get(), 'data.csv')
        with open(csv_filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['No of cycles', 'Crack length in mm','Start pixel','End pixel','Image Name','Aruco Codinates'])
            for value1, value2, value3, value4, value5, value6 in zip(x, y,start_pixel_list,end_pixel_list,file_name_list,aruco_location_list):
                writer.writerow([value1, value2, value3, value4, value5, value6])
        print(f"CSV file '{csv_filepath}' has been created.")
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().place(x=950, y=70)

def get_cycle_num():
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

def get_pixel_rate():
    number = aruco_entry.get()
    try:
        number = float(number)
        pixelmul.set(number)
    except ValueError:
        print("Value Error: Enter numbers only")

def get_squarelength():
    number = sqrlen_entry.get()
    try:
        number = float(number)
        sqrlen.set(number)
    except ValueError:
        print("Value Error: Enter numbers only")

def get_markerlength():
    number = marklen_entry.get()
    try:
        number = float(number)
        marklen.set(number)
    except ValueError:
        print("Value Error: Enter numbers only")

def update_ar(*args):
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


root = tk.Tk()
root.geometry('1500x700')
root.title('Segment Crack')

#Initializing sub-modules
unet = UseModel("weights/best_epoch_weights.pth")
dij = Dijkstra(30)
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
cracklength_list = []
start_pixel_list = []
end_pixel_list = []
file_name_list = []
aruco_location_list = []

# Variable to save output folder path
outputfolder = tk.StringVar()
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