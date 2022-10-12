# from flask import Flask,render_template

# app = Flask(__name__)

# @app.route('/')
# def index():
#     return 'Index Page'

# @app.route('/hello')
# def hello():
#     return 'Hello, eudawrd'

# @app.route('/post/<int:post_id>')
# def show_post(post_id):
#     # show the post with the given id, the id is an integer
#     return f'Post {post_id}'


# @app.route('/index')
# def dex():
#     return render_template('index.html')

# import face_recognition
# from flask import Flask, jsonify, request, redirect

# # You can change this to any folder on your system
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# app = Flask(__name__)


# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# @app.route('/', methods=['GET', 'POST'])
# def upload_image():
#     # Check if a valid image file was uploaded
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return redirect(request.url)

#         file = request.files['file']

#         if file.filename == '':
#             return redirect(request.url)

#         if file and allowed_file(file.filename):
#             # The image file seems valid! Detect faces and return the result.
#             return detect_faces_in_image(file)

#     # If no valid image file was uploaded, show the file upload form:
#     return '''
#     <!doctype html>
#     <title>Is this a picture of Obama?</title>
#     <h1>Upload a picture and see if it's a picture of Obama!</h1>
#     <form method="POST" enctype="multipart/form-data">
#       <input type="file" name="file">
#       <input type="submit" value="Upload">
#     </form>
#     '''


# def detect_faces_in_image(file_stream):
#     # Pre-calculated face encoding of Obama generated with face_recognition.face_encodings(img)
#     known_face_encoding = [-0.09634063,  0.12095481, -0.00436332, -0.07643753,  0.0080383,
#                             0.01902981, -0.07184699, -0.09383309,  0.18518871, -0.09588896,
#                             0.23951106,  0.0986533 , -0.22114635, -0.1363683 ,  0.04405268,
#                             0.11574756, -0.19899382, -0.09597053, -0.11969153, -0.12277931,
#                             0.03416885, -0.00267565,  0.09203379,  0.04713435, -0.12731361,
#                            -0.35371891, -0.0503444 , -0.17841317, -0.00310897, -0.09844551,
#                            -0.06910533, -0.00503746, -0.18466514, -0.09851682,  0.02903969,
#                            -0.02174894,  0.02261871,  0.0032102 ,  0.20312519,  0.02999607,
#                            -0.11646006,  0.09432904,  0.02774341,  0.22102901,  0.26725179,
#                             0.06896867, -0.00490024, -0.09441824,  0.11115381, -0.22592428,
#                             0.06230862,  0.16559327,  0.06232892,  0.03458837,  0.09459756,
#                            -0.18777156,  0.00654241,  0.08582542, -0.13578284,  0.0150229 ,
#                             0.00670836, -0.08195844, -0.04346499,  0.03347827,  0.20310158,
#                             0.09987706, -0.12370517, -0.06683611,  0.12704916, -0.02160804,
#                             0.00984683,  0.00766284, -0.18980607, -0.19641446, -0.22800779,
#                             0.09010898,  0.39178532,  0.18818057, -0.20875394,  0.03097027,
#                            -0.21300618,  0.02532415,  0.07938635,  0.01000703, -0.07719778,
#                            -0.12651891, -0.04318593,  0.06219772,  0.09163868,  0.05039065,
#                            -0.04922386,  0.21839413, -0.02394437,  0.06173781,  0.0292527 ,
#                             0.06160797, -0.15553983, -0.02440624, -0.17509389, -0.0630486 ,
#                             0.01428208, -0.03637431,  0.03971229,  0.13983178, -0.23006812,
#                             0.04999552,  0.0108454 , -0.03970895,  0.02501768,  0.08157793,
#                            -0.03224047, -0.04502571,  0.0556995 , -0.24374914,  0.25514284,
#                             0.24795187,  0.04060191,  0.17597422,  0.07966681,  0.01920104,
#                            -0.01194376, -0.02300822, -0.17204897, -0.0596558 ,  0.05307484,
#                             0.07417042,  0.07126575,  0.00209804]

#     # Load the uploaded image file
#     img = face_recognition.load_image_file(file_stream)
#     # Get face encodings for any faces in the uploaded image
#     unknown_face_encodings = face_recognition.face_encodings(img)

#     face_found = False
#     is_obama = False

#     if len(unknown_face_encodings) > 0:
#         face_found = True
#         # See if the first face in the uploaded image matches the known face of Obama
#         match_results = face_recognition.compare_faces([known_face_encoding], unknown_face_encodings[0])
#         if match_results[0]:
#             is_obama = True

#     # Return the result as json
#     result = {
#         "face_found_in_image": face_found,
#         "is_picture_of_obama": is_obama
#     }
#     return jsonify(result)

# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=5001, debug=True)



from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np

app=Flask(__name__)

# Load a sample picture and learn how to recognize it.
krish_image = face_recognition.load_image_file("Krish/krish.jpg")
krish_face_encoding = face_recognition.face_encodings(krish_image)[0]

# Load a second sample picture and learn how to recognize it.
bradley_image = face_recognition.load_image_file("Bradley/bradley.jpg")
bradley_face_encoding = face_recognition.face_encodings(bradley_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    krish_face_encoding,
    bradley_face_encoding
]
known_face_names = [
    "Krish",
    "Bradly"
]
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

def gen_frames():
    camera = cv2.VideoCapture(0)  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
           
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)
            

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__=='__main__':
    app.run(debug=True)
