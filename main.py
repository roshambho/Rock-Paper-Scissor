# just importing stuff
from cv2 import cv2
import hand_detection_module
from data_generation import num_hand
import pickle
import time
from random import choice
from id_distance import calc_all_distance
from scipy import stats as st
from collections import deque
import numpy as np


# Programme core
model_name = 'hand_model.sav'


# custom function
def rps(num):
    if num == 0: return 'PAPER'
    elif num == 1: return 'ROCK'
    else: return 'SCISSOR'


# Helper function to find out winner
def findout_winner(user_move, Computer_move):

    if user_move == Computer_move:
        return "It's a tie!"

    elif user_move == "ROCK" and Computer_move == "SCISSOR":
        return "You won!"

    elif user_move == "ROCK" and Computer_move == "PAPER":
        return "Computer won"

    elif user_move == "SCISSOR" and Computer_move == "ROCK":
        return "Computer won"

    elif user_move == "SCISSOR" and Computer_move == "PAPER":
        return "You won!"

    elif user_move == "PAPER" and Computer_move == "ROCK":
        return "You won!"

    elif user_move == "PAPER" and Computer_move == "SCISSOR":
        return "Computer won"


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


# helper function to show result screen
def show_winner(user_score, computer_score):
    if user_score > computer_score:
        image = cv2.imread("images/youwin.jpg")

    elif user_score < computer_score:
        image = cv2.imread("images/comwins.png")

    else:
        image = cv2.imread("images/draw.png")

    cv2.imshow("Rock Paper Scissors", image)

    # If enter is pressed.
    k = cv2.waitKey(0)

    # If the user presses 'ENTER' key then return TRUE, otherwise FALSE
    if k == 13:
        return True

    else:
        return False


# variable declaration
font = cv2.FONT_HERSHEY_SIMPLEX
hands = hand_detection_module.HandDetector(max_hands=num_hand)
model = pickle.load(open(model_name, 'rb'))  # Note that hand_model.sav is now loaded in variable 'model'
TIMER = int(3)
cScore = int(0)
uScore = int(0)
smooth_factor = 50  # Instead of working on a single prediction, we will take the mode of 5 predictions
# by using a deque object. This way even if we face a false positive, we would easily ignore it

# Initial deque list will have 'nothing'.
de = deque(['nothing'] * smooth_factor, maxlen=smooth_factor)


# Starting cam
cap = cv2.VideoCapture(2)
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (0, 505), (1280, 570), (0, 0, 0), -1)
    cv2.putText(frame, f"How many rounds would you like to play? (1 to 9)", (50, 550), font, 1.5, (0, 255, 255), 3, cv2.LINE_AA)
    cv2.imshow('Game Start Screen', frame)
    r = cv2.waitKey(25)  # r variable stores no. of rounds of the game
    # print(r) # for debugging purpose
    while 48 < r <= 57:  # Ascii values for 0 & 9
        cv2.destroyAllWindows()
        prev = time.time()

        while TIMER >= 0:
            ret, img = cap.read()
            img = cv2.flip(img, 1)

            # Display countdown on each frame
            # specify the font and draw the countdown using puttext
            cv2.rectangle(img, (140, 490), (1000, 580), (0, 0, 0), -1)
            cv2.putText(img, f"Show your move in: {str(TIMER)}", (200, 550), font, 2, (0, 255, 255), 3, cv2.LINE_AA)
            cv2.imshow('Countdown', img)
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
                success, moveImg = cap.read()  # Capture the images from camera for 50 times (loop)
                moveImg = cv2.flip(moveImg, 1)
                image, list_of_landmarks = hands.find_hand_landmarks(moveImg, draw_landmarks=True)
                height, width, _ = image.shape
                all_distance = calc_all_distance(height, width, list_of_landmarks)
                prediction = rps(model.predict([all_distance])[0])
                de.appendleft(prediction)
                i += 1
            try:
                mode_of_prediction = st.mode(de)[0][0]

            except:
                print('Stats error')
                continue
            cv2.rectangle(image, (0, 0), (1280, 40), (0, 0, 0), -1)
            image = cv2.putText(image, f"Your move: {mode_of_prediction}", (700, 30), font, 1, (0, 255, 0), 3)

            user_move = mode_of_prediction
            computer_move = choice(['ROCK', 'PAPER', 'SCISSOR'])

            winner = findout_winner(user_move, computer_move)
            image = cv2.putText(image, f"Computer's move: {computer_move}", (20, 30), font, 1, (0, 255, 0), 3)

            # Adding black background in bottom3
            cv2.rectangle(image, (0, 550), (1280, 720), (0, 0, 0), -1)

            image = cv2.putText(image, f"Result: {winner}", (450, 600), font, 1.5, (0, 255, 0), 3)
            # image = cv2.putText(image, f"Press '2n' to start next round", (150, 600), font, 2, (0, 255, 0), 3)
            if winner == "You won!":
                uScore += 1
            elif winner == "Computer won":
                cScore += 1

            image = cv2.putText(image, f"Your Score: {uScore}", (800, 650), font, 1, (0, 255, 0), 3)
            image = cv2.putText(image, f"Computer's  Score: {cScore}", (100, 650), font, 1, (0, 255, 0), 3)
            image = cv2.putText(image, f"Press 'Enter' for next round", (100, 700), font, 1, (0, 255, 0), 3)
            image = cv2.putText(image, f"Developed by Aishwary Shukla", (950, 700), font, 0.6, (0, 255, 0), 2)
            display_computer_move(computer_move, image)  # Function call

            r -= 1
            print(r)
            cv2.imshow('Game', image)
            n = cv2.waitKey(0)
            if n == 13:
                cv2.destroyAllWindows()
            if r == 48:
                play_again = show_winner(uScore, cScore)

                if play_again:
                    uScore, cScore = 0, 0

        TIMER = int(3)

    else:
        continue

# Close the camera and destroy all windows.
cap.release()
cv2.destroyAllWindows()
