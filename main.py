import cv2
import hand_detection_module
from data_generation import num_hand
import pickle
import time
from random import choice
from id_distance import calc_all_distance
from scipy import stats as st
from collections import deque
import numpy as np
import pandas as pd
import gif2numpy

def video_render(video_path):
    vid_capture = cv2.VideoCapture(video_path)
     
    if (vid_capture.isOpened() == False):
      print("Error opening the video file")
    # Read fps and frame count
    else:
        # Get frame rate information
        # You can replace 5 with CAP_PROP_FPS as well, they are enumerations
        fps = vid_capture.get(5)
        print('Frames per second : ', fps,'FPS')
         
        # Get frame count
        # You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are enumerations
        frame_count = vid_capture.get(7)
        print('Frame count : ', frame_count)
         
    while(vid_capture.isOpened()):
        # vid_capture.read() methods returns a tuple, first element is a bool
        # and the second is frame
        ret, frame = vid_capture.read()
        if ret == True:
            cv2.imshow('Frame',frame)
            # 20 is in milliseconds, try to increase the value, say 50 and observe
            key = cv2.waitKey(20)
         
            if key == ord('q'):
                break
        else:
            break
    vid_capture.release()
    cv2.destroyAllWindows()


np_frames, extensions, image_specifications = gif2numpy.convert("images/trump.gif")
print(len(np_frames))
print(image_specifications['Image Size'])
"""
for i in range(len(np_frames)):
    cv2.imshow("np_image", np_frames[i])
    cv2.waitKey(100)
    #if k == 27:
    #    break
    cv2.destroyWindow("np_image")
exit()
"""



# Heart of the game
model_name = 'hand_model.sav'

countdown_time = 10
font = cv2.FONT_HERSHEY_SIMPLEX
hands = hand_detection_module.HandDetector(max_hands=num_hand)
model = pickle.load(open(model_name, 'rb')) # Note that hand_model.sav is now loaded in variable 'model'
TIMER = int(countdown_time)
cScore = int(0)
uScore = int(0)
# Instead of working on a single prediction, we will take the mode of 10 predictions
smooth_factor = 5  # by using a deque object. This way even if we face a false positive, we would easily ignore it
# reduce the smooth factor to reduce lag for real-time detection


def rps(num):
    if num == 0:
        return 'PAPER'
    elif num == 1:
        return 'ROCK'
    else:
        return 'SCISSOR'

def findout_winner(user_move, computer_move):
    # Dictionary to store the winning combinations
    winning_combinations = {
        'ROCK': 'SCISSOR',
        'PAPER': 'ROCK',
        'SCISSOR': 'PAPER'
    }
    if user_move == computer_move:
        return "It's a tie!"
    elif winning_combinations[user_move] == computer_move:
        return "You won!"
    else:
        return "Computer won"


def check_selection(user_move, question): #, question <-- to ADD
    # Dictionary to store the winning combinations
    gesture_to_options = {
        'ROCK': 'A',
        'PAPER': 'B',
        'SCISSOR': 'C',
    }
    selected_option = gesture_to_options[user_move]
    correct_option = correct_option_mapping[question]
    if selected_option == correct_option:
        return f"Correct :)"
    else:
        return "Incorrect :/"
    # if user_move == computer_move:
    #     return "It's a tie!"
    # elif winning_combinations[user_move] == computer_move:
    #     return "You won!"
    # else:
    #     return "Computer won"


# helper function to show computer's hand
def display_computer_move(computer_move_name, image):
    icon = cv2.imread("images/{}.png".format(computer_move_name), 1)
    icon = cv2.resize(icon, (300, 140))

    # This is the portion which we are going to replace with the icon image
    roi = image[200:340, 50:350]

    # Get binary mask from the transparent image, 4th channel is the alpha channel
    mask = icon[:, :, -1]

    # Making the mask completely binary (black & white)
    mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]

    # Store the normal bgr image
    icon_bgr = icon[:, :, :3]

    # Now combine the foreground of the icon with background of ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
    img2_fg = cv2.bitwise_and(icon_bgr, icon_bgr, mask=mask)
    combined = cv2.add(img1_bg, img2_fg)
    image[200:340, 50:350] = combined

    return image
#def display_kbc_bar
# Define a list of questions
questions = [
    "Where is Isha yoga center located?",
    "What's the ratio of Indian:Overseas applicants ?",
    "What's on top of vanashri?"
]

options = {
"Where is Isha yoga center located?":{'A': 'Chennai ','B': 'Mumbai ','C': 'Coimbatore'},
"What's the ratio of Indian:Overseas applicants ?":{'A': '1:5','B':'5:1','C': '3:7 '},
"What's on top of vanashri?":{'A':'Monster','B':'Flower','C': 'Om'}
}

correct_option_mapping = {
"Where is Isha yoga center located?":'C',
"What's the ratio of Indian:Overseas applicants ?":'B',
"What's on top of vanashri?":'A'
}

# Initialize the current question

questions_selected = False 

# Create a function to display the current question and options
def display_question_and_options(frame, qn_no, qn_count):
    question = questions[qn_no]
    cv2.putText(frame, f"{qn_count + 1}: {question}", (
        500, 1500), font, 4, (255, 255, 255), 4, cv2.LINE_AA)
    display_options(frame, question)

# Create a function to display the current question
#def display_question(frame):
#    question = questions[current_question]
#    cv2.putText(frame, f"Question {current_question + 1}: {question}", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

# Create a function to display options on the camera screen
def display_options(frame, question):
    y = 1750  # Y-coordinate for the first option
    # y = 50
    """
    for option_key, option_text in options[question].items():
        #cv2.rectangle(frame, (0, y-70), (1200, y+50), (0, 0, 0), -1)
        cv2.putText(frame, f"Option {option_key}: {option_text}", (500, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        # cv2.putText(frame, f"Option {option_key}: {option_text}", (
        #     200, y), font, 2, (0, 255, 255), 3, cv2.LINE_AA)
        y += 100  # Adjust the vertical position for the next option
    """
    optionA = options[question]['A']
    optionB = options[question]['B']
    optionC = options[question]['C']
    cv2.putText(frame, f"{optionA}", (700, 1790), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
    cv2.putText(frame, f"{optionB}", (700, 2030), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 3)
    cv2.putText(frame, f"{optionC}", (2500, 1790), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255),3)

# helper function to show result screen
def show_winner():
    
    image = cv2.imread("images/youwin.jpg")
    image = cv2.resize(image, (3840, 2160), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("Game over", image)

    # If enter is pressed.
    k = cv2.waitKey(0)

    # If the user presses 'ENTER' key then return TRUE, otherwise FALSE
    if k == 13:
        return True

    else:
        return False


# Initial deque list will have 'nothing'.
de = deque(['nothing'] * smooth_factor, maxlen=smooth_factor)
#kbc_template = cv2.imread("images/kbc.jpeg")
#kbc_template = cv2.imread("images/kbc_black.png")
kbc_template = cv2.imread("images/kbc_with_rps.jpeg")
kbc_x1 = 0
kbc_x2 = 3840
kbc_y1 = 2160-850
kbc_y2 = 2160
kbc_template = cv2.resize(kbc_template, (3840, 850), interpolation=cv2.INTER_CUBIC)
# Starting cam
#video_render('images/telepoorte_fnl.mp4')
cap = cv2.VideoCapture(0)
feature_names = [f'feature{i}' for i in range(210)]  # Create feature names
while True:
    #start_image = cv2.imread("images/youwin.jpg")
    #cv2.namedWindow("Display2", cv2.WINDOW_NORMAL)
    #cv2.imshow('Display2', start_image)
    #key = cv2.waitKey(5000)#pauses for 5 seconds before fetching next image
    #break
    #video_render('images/telepoorte_fnl.mp4')
    #im = cv2.imread("images/are_you_ready_1.jpeg")
    im = cv2.imread("images/are_you_ready_2.webp")

    im = cv2.resize(im, (3840, 2160), interpolation=cv2.INTER_LINEAR)
    #cv2.namedWindow("foo", cv2.WINDOW_NORMAL)
    #cv2.setWindowProperty("foo", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("foo", im)
    k = cv2.waitKey(33)
    if k==27:    # Esc key to stop
        break
    elif k==-1:  # normally -1 returned,so don't print it
        continue
    else:
        print(k)
        cv2.destroyWindow("foo")
    #cv2.waitKey()
    
    #
    #if key == 27:#if ESC is pressed, exit loop
    #    cv2.destroyAllWindows()
    #    break
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    #cv2.rectangle(frame, (0, 505), (1280, 570), (0, 0, 0), -1)
    #cv2.putText(frame, f"How many rounds would you like to play? (1 to 9)",
    #            (50, 550), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)
    

    #r = cv2.waitKey(25)  # r variable stores the number of rounds of the game
    r = 51 #r variable stores the number of rounds of the game,r = 49 maps to 1 round
    #cv2.imshow('Game Start Screen', frame)
    
    
    ######## insert gesture checks ######
    
    #cv2.waitKey(5000)

    #print(cv2.getWindowImageRect('Game Start Screen'))


    #print("rounds",r) # for debugging purpose
    total_questions = r-48 # r = 49 maps to 1 round hence 1 question
    #change total_questions to be global variable since the number of rounds can be lesser than total questions in bank
    question_sequence = np.random.permutation(total_questions).tolist()
    #print(question_sequence)
    qn_count = 0
    # wait for user to choose no of rounds between 0 to 9
    while 48 < r <= 57:  # Ascii values for 0 & 9
        cv2.destroyAllWindows()
        
        # # Create a new frame/rectangle for displaying question and options
        # cv2.rectangle(frame, (0, 0), (1280, 720), (0, 0, 0), -1)

        prev = time.time()
        print("qn count",qn_count)
        qn_no = question_sequence[qn_count]
        question = questions[qn_no]
        print("qn no",qn_no)
        while TIMER >= 0:
        ########## REAL-TIME RECOGNITION START ##########
           
            i = 0
            while i < smooth_factor:
                success, moveImg = cap.read()
                moveImg = cv2.flip(moveImg, 1)
                image, list_of_landmarks = hands.find_hand_landmarks(
                    moveImg, draw_landmarks=True)
                height, width, _ = image.shape
                all_distance = calc_all_distance(
                    height, width, list_of_landmarks)
                # Convert all_distance to a DataFrame and set column names
                all_distance_df = pd.DataFrame(
                    [all_distance], columns=feature_names)

                # Use the DataFrame for prediction
                prediction = rps(model.predict(all_distance_df)[0])
                de.appendleft(prediction)
                i += 1

            de_list = list(de)  # Convert deque to list
            # Create DataFrame from list
            de_df = pd.DataFrame(de_list, columns=['prediction'])
            mode_of_prediction = de_df['prediction'].mode()[0]
            image = cv2.resize(image, (3840, 2160), interpolation=cv2.INTER_LINEAR)

            cv2.rectangle(image, (0, 0), (3840, 200), (0, 0, 0), -1)
            image = cv2.putText(image, f"Your gesture:", (1250, 140), font, 4, (0, 165, 255), 5)
            user_move = mode_of_prediction
            image = cv2.putText(image, f"{mode_of_prediction}", (2100, 140), font, 4, (255, 255, 255), 5)

            #print("here")
            ## superimpose kbc template
            image[kbc_y1:kbc_y2, kbc_x1:kbc_x2] = kbc_template
            cv2.imshow('Countdown',image)
            ########### REAL-TIME RECOGNITION END ########
        
            #ret, image = cap.read()
            #image = cv2.flip(image, 1)

            # Display countdown on each frame
            # specify the font and draw the countdown using puttext
            #

            display_question_and_options(image, qn_no, qn_count)

            #cv2.rectangle(image, (0, 900), (1919, 1000), (0, 0, 0), -1)
            #cv2.rectangle(image, (600, 1000), (800, 1080), (0, 0, 0), -1)
            #image = cv2.putText(image, f"Show a gesture (Rock/Paper/Scissor) to select an option", (
            #    100, 950), font, 1.8, (0, 255, 0), 3, cv2.LINE_AA) #TODO: change font
            image = cv2.putText(image, f"{str(TIMER)}", (3640, 140), font, 4, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.imshow('Countdown', image)
            cv2.waitKey(25)

            

            # current time
            cur = time.time()

            # Update and keep track of Countdown
            # if time elapsed is one second
            # than decrease the counter
            if cur - prev >= 1:
                prev = cur
                TIMER = TIMER - 1
            
        else:  # when timer is up
            cv2.destroyAllWindows()
            i = 0
            while i < smooth_factor:
                success, moveImg = cap.read()
                moveImg = cv2.flip(moveImg, 1)
                image, list_of_landmarks = hands.find_hand_landmarks(
                    moveImg, draw_landmarks=True)
                height, width, _ = image.shape
                all_distance = calc_all_distance(
                    height, width, list_of_landmarks)
                # Convert all_distance to a DataFrame and set column names
                all_distance_df = pd.DataFrame(
                    [all_distance], columns=feature_names)

                # Use the DataFrame for prediction
                prediction = rps(model.predict(all_distance_df)[0])
                de.appendleft(prediction)
                i += 1

            de_list = list(de)  # Convert deque to list
            # Create DataFrame from list
            de_df = pd.DataFrame(de_list, columns=['prediction'])
            mode_of_prediction = de_df['prediction'].mode()[0]
            image = cv2.resize(image, (3840, 2160), interpolation=cv2.INTER_LINEAR)
            
            cv2.rectangle(image, (0, 0), (3840, 200), (0, 0, 0), -1)
            image = cv2.putText(image, f"Your gesture:", (1250, 140), font, 4, (0, 165, 255), 5)
            user_move = mode_of_prediction
            image = cv2.putText(image, f"{mode_of_prediction}", (2100, 140), font, 4, (255, 255, 255), 5)
            #winner = findout_winner(user_move, computer_move)

            option_selected_correctly = check_selection(user_move,question) # RRAM


            #image = cv2.putText(
            #    image, f"Computer's move: {computer_move}", (20, 30), font, 1, (0, 255, 0), 3)

            # Adding black background in bottom3
            cv2.rectangle(image, (0, 1560), (3840, 2160), (0, 0, 0), -1)

            image = cv2.putText(image, f"{option_selected_correctly}", (1600, 1690), font, 4, (0, 255, 0), 4)
            # image = cv2.putText(image, f"Press '2n' to start next round", (150, 600), font, 2, (0, 255, 0), 3)
            if option_selected_correctly != "Incorrect :/":
                uScore += 100
            
            image = cv2.putText(image, f"Your Score:", (1600, 1860), font, 3, (0, 165, 255), 3)
            image = cv2.putText(image, f" {uScore}", (2100, 1860), font, 3, (255, 255, 255), 3)
            #image = cv2.putText(image, f"Press 'Enter' for next round", (200, 2060), font, 3, (0, 255, 0), 3)
            #image = cv2.putText(image, f"Attention is all you need", (3000, 2060), font, 2, (0, 255, 0), 3)
            #display_computer_move(computer_move, image)  # Function call
            
            qn_count += 1
            r -= 1
            print(r-48)
            #cv2.imshow('Game', image)
            for i in range(len(np_frames)-1):
                image[1500:1787, 2000:2480] = np_frames[i]
                #cv2.imshow("np_image", np_frames[i])
                cv2.imshow("np_image", image)
                cv2.waitKey(100)
                #if k == 27:
                #    break
                cv2.destroyWindow("np_image")
            cv2.imshow("np_image", image)
            n = cv2.waitKey(3000)
            if n == 13:
                cv2.destroyAllWindows()
            if r == 48:
                play_again = show_winner()

                if play_again:
                    uScore, cScore = 0, 0

        TIMER = int(countdown_time)
        

    else:
        continue

# Close the camera and destroy all windows.
cap.release()
cv2.destroyAllWindows()
