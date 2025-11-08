from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import io
from PIL import Image
import math
import pyttsx3
from gtts import gTTS
import pygame
import tempfile
import os
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
from string import ascii_uppercase
import enchant
import threading
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sign_language_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize components
ddd = enchant.Dict("en-US")
hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)
model = load_model('cnn8grps_rad1_model.h5')

# Initialize pygame mixer for audio playback
pygame.mixer.init()

offset = 29

# Global variables for state management
current_symbol = ""
current_sentence = ""
suggestions = ["", "", "", ""]
ten_prev_char = [" "] * 10
count = -1
prev_char = ""
last_next_time = 0
last_character_time = 0
last_backspace_time = 0

# Gesture smoothing
gesture_history = []  # Store last 5 gesture predictions
smoothing_window = 3  # Number of consecutive frames needed for stable gesture
current_gesture_count = 0
last_stable_gesture = ""

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {'message': 'Connected to Sign Language Recognition System'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('process_frame')
def handle_frame(data):
    global current_symbol, current_sentence, suggestions, ten_prev_char, count, prev_char, last_next_time, last_character_time, last_backspace_time
    global gesture_history, smoothing_window, current_gesture_count, last_stable_gesture
    
    try:
        # Decode base64 image
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Flip frame horizontally
        cv2image = cv2.flip(frame, 1)
        
        # Find hands
        hands = hd.findHands(cv2image, draw=False, flipType=True)
        
        if hands and len(hands) > 0:
            hand = hands[0]
            map_data = hand[0]
            x, y, w, h = map_data['bbox']
            image_crop = cv2image[y - offset:y + h + offset, x - offset:x + w + offset]
            
            white = cv2.imread("white.jpg")
            
            if image_crop.size > 0:
                handz = hd2.findHands(image_crop, draw=False, flipType=True)
                
                if handz and len(handz) > 0:
                    hand_data = handz[0]
                    handmap = hand_data[0]
                    pts = handmap['lmList']
                    
                    # Draw hand skeleton
                    os = ((400 - w) // 2) - 15
                    os1 = ((400 - h) // 2) - 15
                    
                    # Draw finger connections
                    for t in range(0, 4, 1):
                        cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), 
                                (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                    for t in range(5, 8, 1):
                        cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), 
                                (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                    for t in range(9, 12, 1):
                        cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), 
                                (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                    for t in range(13, 16, 1):
                        cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), 
                                (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                    for t in range(17, 20, 1):
                        cv2.line(white, (pts[t][0] + os, pts[t][1] + os1), 
                                (pts[t + 1][0] + os, pts[t + 1][1] + os1), (0, 255, 0), 3)
                    
                    # Draw palm connections
                    cv2.line(white, (pts[5][0] + os, pts[5][1] + os1), (pts[9][0] + os, pts[9][1] + os1), (0, 255, 0), 3)
                    cv2.line(white, (pts[9][0] + os, pts[9][1] + os1), (pts[13][0] + os, pts[13][1] + os1), (0, 255, 0), 3)
                    cv2.line(white, (pts[13][0] + os, pts[13][1] + os1), (pts[17][0] + os, pts[17][1] + os1), (0, 255, 0), 3)
                    cv2.line(white, (pts[0][0] + os, pts[0][1] + os1), (pts[5][0] + os, pts[5][1] + os1), (0, 255, 0), 3)
                    cv2.line(white, (pts[0][0] + os, pts[0][1] + os1), (pts[17][0] + os, pts[17][1] + os1), (0, 255, 0), 3)
                    
                    # Draw landmarks
                    for i in range(21):
                        cv2.circle(white, (pts[i][0] + os, pts[i][1] + os1), 2, (0, 0, 255), 1)
                    
                    # Predict gesture
                    prediction_result = predict_gesture(white, pts)
                    predicted_char = prediction_result['character']
                    current_time = time.time()
                    
                    # Apply gesture smoothing
                    gesture_history.append(predicted_char)
                    if len(gesture_history) > 5:  # Keep only last 5 predictions
                        gesture_history.pop(0)
                    
                    # Check if we have enough consistent predictions
                    if len(gesture_history) >= smoothing_window:
                        # Count consecutive same gestures
                        recent_gestures = gesture_history[-smoothing_window:]
                        if all(g == predicted_char for g in recent_gestures):
                            # Stable gesture detected
                            if predicted_char != last_stable_gesture:
                                current_symbol = predicted_char
                                last_stable_gesture = predicted_char
                                current_gesture_count = 0
                            else:
                                current_gesture_count += 1
                        else:
                            # Gesture not stable yet - keep previous character
                            emit('recognition_result', {
                                'character': current_symbol,
                                'sentence': current_sentence,
                                'suggestions': suggestions,
                                'skeleton': f"data:image/jpeg;base64,{skeleton_b64}"
                            })
                            return
                    else:
                        # Not enough history yet - keep previous character
                        emit('recognition_result', {
                            'character': current_symbol,
                            'sentence': current_sentence,
                            'suggestions': suggestions,
                            'skeleton': f"data:image/jpeg;base64,{skeleton_b64}"
                        })
                        return
                    
                    # Handle sentence updates with timing delays
                    if prediction_result['action'] == 'next' and prev_char != 'next':
                        # Check if enough time has passed since last next gesture (1 second delay)
                        if current_time - last_next_time > 1.0:
                            # Check if there was a character input between next gestures
                            if current_time - last_character_time < 5.0:  # Character must be within 5 seconds
                                # Add the previous character to sentence
                                if count >= 2 and ten_prev_char[(count - 2) % 10] != 'next':
                                    if ten_prev_char[(count - 2) % 10] == 'backspace':
                                        current_sentence = current_sentence[:-1] if current_sentence else ""
                                    else:
                                        if ten_prev_char[(count - 2) % 10] != 'backspace':
                                            current_sentence += ten_prev_char[(count - 2) % 10]
                                elif count >= 0 and ten_prev_char[(count - 0) % 10] != 'next':
                                    if ten_prev_char[(count - 0) % 10] != 'backspace':
                                        current_sentence += ten_prev_char[(count - 0) % 10]
                                last_next_time = current_time
                    
                    elif prediction_result['action'] == 'add_space' and prev_char != 'add_space':
                        current_sentence += "  "  # Double space for space gesture
                    
                    elif prediction_result['action'] == 'backspace' and prev_char != 'backspace':
                        # Check if enough time has passed since last backspace (0.5 second delay)
                        if current_time - last_backspace_time > 0.5:
                            current_sentence = current_sentence[:-1] if current_sentence else ""
                            last_backspace_time = current_time
                    
                    # Track when a character (not special gesture) is detected
                    if prediction_result['action'] == 'add_char':
                        last_character_time = current_time
                    
                    # Update tracking variables
                    prev_char = prediction_result['action']
                    count += 1
                    ten_prev_char[count % 10] = current_symbol
                    
                    # Get word suggestions
                    if current_sentence.strip():
                        last_word = current_sentence.split()[-1] if current_sentence.split() else ""
                        if last_word:
                            try:
                                suggestions_list = ddd.suggest(last_word)
                                suggestions = suggestions_list[:4] if len(suggestions_list) >= 4 else suggestions_list + [""] * (4 - len(suggestions_list))
                            except:
                                suggestions = ["", "", "", ""]
                        else:
                            suggestions = ["", "", "", ""]
                    else:
                        suggestions = ["", "", "", ""]
                    
                    # Convert skeleton to base64 for display
                    _, buffer = cv2.imencode('.jpg', white)
                    skeleton_b64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Emit results
                    emit('recognition_result', {
                        'character': current_symbol,
                        'sentence': current_sentence,
                        'suggestions': suggestions,
                        'skeleton': f"data:image/jpeg;base64,{skeleton_b64}"
                    })
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        emit('error', {'message': str(e)})

@socketio.on('speak_text')
def handle_speak():
    global current_sentence
    try:
        if not current_sentence.strip():
            emit('speak_status', {'message': 'No text to speak'})
            return
            
        # Try Google Text-to-Speech first (much better quality)
        try:
            tts = gTTS(text=current_sentence, lang='en', slow=False)
            
            # Create temporary file for audio
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                tts.save(tmp_file.name)
                
                # Play the audio using pygame
                pygame.mixer.music.load(tmp_file.name)
                pygame.mixer.music.play()
                
                # Wait for playback to complete
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)
                
                # Clean up temporary file
                os.unlink(tmp_file.name)
            
            emit('speak_status', {'message': 'Text spoken successfully (Google TTS)'})
            
        except Exception as gtts_error:
            # Fallback to pyttsx3 if Google TTS fails (offline mode)
            print(f"Google TTS failed, using fallback: {gtts_error}")
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)  # Slightly faster for better experience
            voices = engine.getProperty("voices")
            if voices:
                engine.setProperty("voice", voices[0].id)
            
            engine.say(current_sentence)
            engine.runAndWait()
            engine.stop()
            
            emit('speak_status', {'message': 'Text spoken successfully (Fallback TTS)'})
            
    except Exception as e:
        emit('error', {'message': f'Speech error: {str(e)}'})

@socketio.on('clear_text')
def handle_clear():
    global current_sentence
    current_sentence = ""
    emit('clear_result', {'sentence': current_sentence})

@socketio.on('use_suggestion')
def handle_suggestion(data):
    global current_sentence
    suggestion = data['suggestion']
    if suggestion:
        # Replace last word with suggestion
        words = current_sentence.split()
        if words:
            words[-1] = suggestion
            current_sentence = " ".join(words)
        else:
            current_sentence = suggestion
        
        emit('suggestion_applied', {'sentence': current_sentence})

def distance(x, y):
    return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))


def predict_gesture(white_image, pts):
    """Predict gesture using CNN model and hand landmark analysis"""
    white = white_image.reshape(1, 400, 400, 3)
    prob = np.array(model.predict(white)[0], dtype='float32')
    ch1 = np.argmax(prob, axis=0)
    prob[ch1] = 0
    ch2 = np.argmax(prob, axis=0)
    prob[ch2] = 0
    ch3 = np.argmax(prob, axis=0)
    prob[ch3] = 0
    
    pl = [ch1, ch2]
    
    # Classification rules for gesture recognition
    # Comprehensive implementation of gesture classification logic
    
    # condition for [Aemnst]
    l = [[5,2],[5,3],[3,5],[3,6],[3,0],[3,2],[6,4],[6,1],[6,2],[6,6],[6,7],[6,0],[6,5],[4,1],[1,0],[1,1],[6,3],[1,6],[5,6],[5,1],[4,5],[1,4],[1,5],[2,0],[2,6],[4,6],[1,0],[5,7],[1,6],[6,1],[7,6],[2,5],[7,1],[5,4],[7,0],[7,5],[7,2]]
    if pl in l:
        if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 0

    # condition for [o][s]
    l = [[2,2],[2,1]]
    if pl in l:
        if (pts[5][0] < pts[4][0]):
            ch1 = 0

    # condition for [c0][aemnst]
    l = [[0,0],[0,6],[0,2],[0,5],[0,1],[0,7],[5,2],[7,6],[7,1]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[0][0] > pts[8][0] and pts[0][0] > pts[4][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0]) and pts[5][0] > pts[4][0]:
            ch1 = 2

    # condition for [c0][aemnst]
    l = [[6,0],[6,6],[6,2]]
    pl = [ch1, ch2]
    if pl in l:
        if distance(pts[8], pts[16]) < 52:
            ch1 = 2

    # condition for [gh][bdfikruvw]
    l = [[1,4],[1,5],[1,6],[1,3],[1,0]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[6][1] > pts[8][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1] and pts[0][0] < pts[8][0] and pts[0][0] < pts[12][0] and pts[0][0] < pts[16][0] and pts[0][0] < pts[20][0]:
            ch1 = 3

    # con for [gh][l]
    l = [[4,6],[4,1],[4,5],[4,3],[4,7]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[4][0] > pts[0][0]:
            ch1 = 3

    # con for [gh][pqz]
    l = [[5,3],[5,0],[5,7],[5,4],[5,2],[5,1],[5,5]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[2][1] + 15 < pts[16][1]:
            ch1 = 3

    # con for [l][x]
    l = [[6,4],[6,1],[6,2]]
    pl = [ch1, ch2]
    if pl in l:
        if distance(pts[4], pts[11]) > 55:
            ch1 = 4

    # con for [l][d]
    l = [[1,4],[1,6],[1,1]]
    pl = [ch1, ch2]
    if pl in l:
        if (distance(pts[4], pts[11]) > 50) and (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 4

    # con for [l][gh]
    l = [[3,6],[3,4]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[4][0] < pts[0][0]):
            ch1 = 4

    # con for [l][c0]
    l = [[2,2],[2,5],[2,4]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[1][0] < pts[12][0]):
            ch1 = 4

    # con for [gh][z]
    l = [[3,6],[3,5],[3,4]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and pts[4][1] > pts[10][1]:
            ch1 = 5

    # con for [gh][pq]
    l = [[3,2],[3,1],[3,6]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[4][1] + 17 > pts[8][1] and pts[4][1] + 17 > pts[12][1] and pts[4][1] + 17 > pts[16][1] and pts[4][1] + 17 > pts[20][1]:
            ch1 = 5

    # con for [l][pqz]
    l = [[4,4],[4,5],[4,2],[7,5],[7,6],[7,0]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[4][0] > pts[0][0]:
            ch1 = 5

    # con for [pqz][aemnst]
    l = [[0,2],[0,6],[0,1],[0,5],[0,0],[0,7],[0,4],[0,3],[2,7]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[0][0] < pts[8][0] and pts[0][0] < pts[12][0] and pts[0][0] < pts[16][0] and pts[0][0] < pts[20][0]:
            ch1 = 5

    # con for [pqz][yj]
    l = [[5,7],[5,2],[5,6]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[3][0] < pts[0][0]:
            ch1 = 7

    # con for [l][yj]
    l = [[4,6],[4,2],[4,4],[4,1],[4,5],[4,7]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[6][1] < pts[8][1]:
            ch1 = 7

    # con for [x][yj]
    l = [[6,7],[0,7],[0,1],[0,0],[6,4],[6,6],[6,5],[6,1]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[18][1] > pts[20][1]:
            ch1 = 7

    # condition for [x][aemnst]
    l = [[0,4],[0,2],[0,3],[0,1],[0,6]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[5][0] > pts[16][0]:
            ch1 = 6

    # condition for [yj][x]
    l = [[7,2]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[18][1] < pts[20][1] and pts[8][1] < pts[10][1]:
            ch1 = 6

    # condition for [c0][x]
    l = [[2,1],[2,2],[2,6],[2,7],[2,0]]
    pl = [ch1, ch2]
    if pl in l:
        if distance(pts[8], pts[16]) > 50:
            ch1 = 6

    # con for [l][x]
    l = [[4,6],[4,2],[4,1],[4,4]]
    pl = [ch1, ch2]
    if pl in l:
        if distance(pts[4], pts[11]) < 60:
            ch1 = 6

    # con for [x][d]
    l = [[1,4],[1,6],[1,0],[1,2]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[5][0] - pts[4][0] - 15 > 0:
            ch1 = 6

    # con for [b][pqz]
    l = [[5,0],[5,1],[5,4],[5,5],[5,6],[6,1],[7,6],[0,2],[7,1],[7,4],[6,6],[7,2],[5,0],[6,3],[6,4],[7,5],[7,2]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 1

    # con for [f][pqz]
    l = [[6,1],[6,0],[0,3],[6,4],[2,2],[0,6],[6,2],[7,6],[4,6],[4,1],[4,2],[0,2],[7,1],[7,4],[6,6],[7,2],[7,5],[7,2]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 1

    l = [[6,1],[6,0],[4,2],[4,1],[4,6],[4,4]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 1

    # con for [d][pqz]
    l = [[5,0],[3,4],[3,0],[3,1],[3,5],[5,5],[5,4],[5,1],[7,6]]
    pl = [ch1, ch2]
    if pl in l:
        if ((pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and (pts[2][0] < pts[0][0]) and pts[4][1] > pts[14][1]):
            ch1 = 1

    l = [[4,1],[4,2],[4,4]]
    pl = [ch1, ch2]
    if pl in l:
        if (distance(pts[4], pts[11]) < 50) and (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 1

    l = [[3,4],[3,0],[3,1],[3,5],[3,6]]
    pl = [ch1, ch2]
    if pl in l:
        if ((pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and (pts[2][0] < pts[0][0]) and pts[14][1] < pts[4][1]):
            ch1 = 1

    l = [[6,6],[6,4],[6,1],[6,2]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[5][0] - pts[4][0] - 15 < 0:
            ch1 = 1

    # con for [i][pqz]
    l = [[5,4],[5,5],[5,1],[0,3],[0,7],[5,0],[0,2],[6,2],[7,5],[7,1],[7,6],[7,7]]
    pl = [ch1, ch2]
    if pl in l:
        if ((pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1])):
            ch1 = 1

    # con for [yj][bfdi]
    l = [[1,5],[1,7],[1,1],[1,6],[1,3],[1,0]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[4][0] < pts[5][0] + 15) and ((pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1])):
            ch1 = 7

    # con for [uvr]
    l = [[5,5],[5,0],[5,4],[5,1],[4,6],[4,1],[7,6],[3,0],[3,5]]
    pl = [ch1, ch2]
    if pl in l:
        if ((pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1])) and pts[4][1] > pts[14][1]:
            ch1 = 1

    # con for [w]
    fg = 13
    l = [[3,5],[3,0],[3,6],[5,1],[4,1],[2,0],[5,0],[5,5]]
    pl = [ch1, ch2]
    if pl in l:
        if not (pts[0][0] + fg < pts[8][0] and pts[0][0] + fg < pts[12][0] and pts[0][0] + fg < pts[16][0] and pts[0][0] + fg < pts[20][0]) and not (pts[0][0] > pts[8][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0]) and distance(pts[4], pts[11]) < 50:
            ch1 = 1

    l = [[5,0],[5,5],[0,1]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1]:
            ch1 = 1

    # -------------------------condn for 8 groups ends

    # -------------------------condn for subgroups starts
    if ch1 == 0:
        ch1 = 'S'
        if pts[4][0] < pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0]:
            ch1 = 'A'
        if pts[4][0] > pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0] and pts[4][1] < pts[14][1] and pts[4][1] < pts[18][1]:
            ch1 = 'T'
        if pts[4][1] > pts[8][1] and pts[4][1] > pts[12][1] and pts[4][1] > pts[16][1] and pts[4][1] > pts[20][1]:
            ch1 = 'E'
        if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][0] > pts[14][0] and pts[4][1] < pts[18][1]:
            ch1 = 'M'
        if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][1] < pts[18][1] and pts[4][1] < pts[14][1]:
            ch1 = 'N'

    if ch1 == 2:
        if distance(pts[12], pts[4]) > 42:
            ch1 = 'C'
        else:
            ch1 = 'O'

    if ch1 == 3:
        if (distance(pts[8], pts[12])) > 72:
            ch1 = 'G'
        else:
            ch1 = 'H'

    if ch1 == 7:
        if distance(pts[8], pts[4]) > 42:
            ch1 = 'Y'
        else:
            ch1 = 'J'

    if ch1 == 4:
        ch1 = 'L'

    if ch1 == 6:
        ch1 = 'X'

    if ch1 == 5:
        if pts[4][0] > pts[12][0] and pts[4][0] > pts[16][0] and pts[4][0] > pts[20][0]:
            if pts[8][1] < pts[5][1]:
                ch1 = 'Z'
            else:
                ch1 = 'Q'
        else:
            ch1 = 'P'

    if ch1 == 1:
        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 'B'
        if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 'D'
        if (pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 'F'
        if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 'I'
        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 'W'
        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and pts[4][1] < pts[9][1]:
            ch1 = 'K'
        if ((distance(pts[8], pts[12]) - distance(pts[6], pts[10])) < 8) and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 'U'
        if ((distance(pts[8], pts[12]) - distance(pts[6], pts[10])) >= 8) and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and (pts[4][1] > pts[9][1]):
            ch1 = 'V'
        if (pts[8][0] > pts[12][0]) and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 'R'

    # Check for special gestures
    action = 'add_char'
    if ch1 == 1 or ch1 == 'E' or ch1 == 'S' or ch1 == 'X' or ch1 == 'Y' or ch1 == 'B':
        if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]):
            action = 'add_space'

    if ch1 == 'E' or ch1 == 'Y' or ch1 == 'B':
        if (pts[4][0] < pts[5][0]) and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            action = 'next'

    if pts[0][0] > pts[8][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0] and pts[4][1] < pts[8][1] and pts[4][1] < pts[12][1] and pts[4][1] < pts[16][1] and pts[4][1] < pts[20][1] and pts[4][1] < pts[6][1] and pts[4][1] < pts[10][1] and pts[4][1] < pts[14][1] and pts[4][1] < pts[18][1]:
        action = 'backspace'
    
    return {'character': ch1, 'action': action}

if __name__ == '__main__':
    print("Starting Sign Language Recognition Web Application...")
    print("Open your browser and go to: http://localhost:5000")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
