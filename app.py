#app.py
from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import cv2
from ultralytics import YOLO  # YOLO model library
app = Flask(__name__)
 
#UPLOAD_FOLDER = 'static/uploads'
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'outputs_new')

 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER']=OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

# Load the trained YOLO model globally
print("Loading YOLO model...")
model = YOLO(r'D:\my  stuff\BMS COLLEGE\5THSEM\miniprojectSem5\Website\static\model\best_samraat.pt')
class_names = model.names

# Debug: Print class details
print("Number of classes:", len(class_names))
print("Class labels:", class_names)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/')
def home():
    return render_template('app.html')
 
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_filename = f"processed_{filename}"  # Name for the output image
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        # Save the uploaded file
        file.save(input_path)

        # Process the image
        process_image(input_path, output_path)

        flash('Image successfully uploaded and processed')
        return render_template('app.html', input_filename=filename, output_filename=output_filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)

 
@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='outputs_new/' + filename), code=301)
 
def process_image(input_path, output_path):
    # Make predictions on the input image
    results = model(input_path)

    # Extract prediction details
    boxes = results[0].boxes
    confidences = boxes.conf
    class_ids = boxes.cls.int()
    class_names = model.names

    # Define a normalized color map
    color_map = {k.strip().lower(): v for k, v in {
    "Aluminium foil": (255, 0, 0),  # Red
    "Bottle cap": (0, 255, 0),  # Green
    "Bottle": (0, 0, 255),  # Blue
    "Broken glass": (255, 165, 0),  # Orange
    "Can": (255, 255, 0),  # Yellow
    "Carton": (0, 255, 255),  # Cyan
    "Cigarette": (128, 0, 128),  # Purple
    "Cup": (255, 105, 180),  # Hot Pink
    "Lid": (255, 69, 0),  # Red-Orange
    "Other litter": (128, 128, 128),  # Gray
    "Other plastic": (0, 255, 0),  # Green
    "Paper": (0, 128, 0),  # Dark Green
    "Plastic bag - wrapper": (255, 182, 193),  # Light Pink
    "Plastic container": (186, 85, 211),  # Medium Orchid
    "Pop tab": (255, 140, 0),  # Dark Orange
    "Straw": (72, 61, 139),  # Dark Slate Blue
    "Styrofoam piece": (0, 255, 255),  # Cyan
    "Unlabeled litter": (169, 169, 169),  # Dark Gray
    }.items()}

    # Read the image
    image = cv2.imread(input_path)

    # Iterate through predictions
    for i, box in enumerate(boxes.xyxy):
        x1, y1, x2, y2 = map(int, box.tolist())
        label = class_names[int(class_ids[i])].strip().lower()
        confidence = confidences[i].item()
        color = color_map.get(label, (255, 255, 255))  # Default to white if label not in color_map

        # Debugging: Check if colors and labels are correct
        print(f"Predicted label: {label}, Confidence: {confidence:.2f}, Assigned color: {color}")

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Add label and confidence
        label_text = f"{label} ({confidence:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_size = cv2.getTextSize(label_text, font, font_scale, 1)[0]
        text_x = x1
        text_y = y1 - 10 if y1 - 10 > 10 else y1 + text_size[1]
        cv2.putText(image, label_text, (text_x, text_y), font, font_scale, color, 1, cv2.LINE_AA)

    # Save the processed image
    cv2.imwrite(output_path, image)
    cv2.destroyAllWindows()
    #display_image(output_path)

if __name__ == "__main__":
    app.run(debug=True)