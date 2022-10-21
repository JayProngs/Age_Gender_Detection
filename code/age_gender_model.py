import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN

# load face detector
detector = MTCNN()

# Load model
model = tf.keras.models.load_model('Age_gender_detection.h5')


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def detect_face(img):
    mt_res = detector.detect_faces(img)
    return_res = []

    for face in mt_res:
        x, y, width, height = face['box']
        center = [x + (width / 2), y + (height / 2)]
        max_border = max(width, height)

        # center alignment
        left = max(int(center[0] - (max_border / 2)), 0)
        right = max(int(center[0] + (max_border / 2)), 0)
        top = max(int(center[1] - (max_border / 2)), 0)
        bottom = max(int(center[1] + (max_border / 2)), 0)

        # crop the face
        center_img_k = img[top:top + max_border,
                       left:left + max_border, :]

        # Create the predictions
        images = []
        image = center_img_k
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (48, 48))
        images.append(image)
        images_f = np.array(images)
        images_f_2 = images_f / 255
        images_f_2.shape
        image_test = images_f_2[0]
        pred_1 = model.predict(np.array([image_test]))
        age = int(np.round(pred_1[1][0]))
        gender = int(np.round(pred_1[0][0]))

        # output to the cv2
        return_res.append([top, right, bottom, left, gender, age])

    return return_res


# Get a reference to webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces in the current frame of video
    face_locations = detect_face(rgb_frame)

    # Display the results
    for top, right, bottom, left, gender_preds, age_preds in face_locations:
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        gender_text = 'Female' if gender_preds == 1 else 'Male'
        cv2.putText(frame, 'Gender: {}'.format(gender_text), (left, top - 10),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5, (0, 255, 0), 1)
        cv2.putText(frame, 'Age: {}'.format(age_preds), (left, top - 25), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                    (0, 255, 0), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'x' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
