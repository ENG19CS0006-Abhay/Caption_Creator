from tkinter import *
from tkinter import filedialog
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image, ImageTk

# Create Tkinter window
window = Tk()
window.title("Image Captioning")
window.geometry("500x600")

# Load models
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define variables
image_path = None
output_text = StringVar()
output_text.set("Please select an image.")

# Function to select image file
def select_image():
    global image_path
    image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    image = Image.open(image_path)
    image = image.resize((300, 300))
    image = ImageTk.PhotoImage(image)
    image_label.configure(image=image)
    image_label.image = image
    output_text.set("Selected image: " + image_path)

# Function to generate captions
def generate_captions():
    global image_path
    if image_path is None:
        output_text.set("Please select an image.")
        return
    try:
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert(mode="RGB")
        image = image.resize((224, 224))
        pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        gen_kwargs = {"max_length": 16, "num_beams": 10, "num_return_sequences": 3}
        output_ids = model.generate(pixel_values, **gen_kwargs)
        preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        output_text.set("\n".join(preds))
    except:
        output_text.set("Error generating captions.")

def generate_captions1():
    global image_path
    if image_path is None:
        output_text.set("Please select an image.")
        return
    try:
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert(mode="RGB")
        image = image.resize((224, 224))
        pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)
        gen_kwargs = {"max_length": 16, "num_beams": 10, "num_return_sequences": 1}
        output_ids = model.generate(pixel_values, **gen_kwargs)
        preds = tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        output_text.set(preds)
    except:
        output_text.set("Error generating captions.")


# Create GUI elements
image_label = Label(window)
image_label.pack(pady=10)
select_image_button = Button(window, text="Select Image", command=select_image)
select_image_button.pack(pady=10)
generate_button = Button(window, text="Multiple Captions", command=generate_captions)
generate_button.pack(pady=10)
generate_button = Button(window, text="Single Captions", command=generate_captions1)
generate_button.pack(pady=10)
output_label = Label(window, textvariable=output_text)
output_label.pack(pady=10)

# Run the Tkinter event loop
window.mainloop()
