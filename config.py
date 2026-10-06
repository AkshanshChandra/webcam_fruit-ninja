"""Tunable constants for the game. Distances are expressed for a 720px-tall
canvas and scaled at runtime, so the game feels the same at any resolution."""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Window / canvas -------------------------------------------------------
WINDOW_NAME = "Fruit Ninja"
CANVAS_HEIGHT = 720            # internal render height; camera frames are resized to this
CAMERA_REQUEST_SIZE = (1280, 720)
CAMERA_FPS = 30

GAME_TITLE = "FRUIT NINJA"
GAME_SUBTITLE = "WEBCAM EDITION"

# --- Files -----------------------------------------------------------------
LEADERBOARD_FILE = os.path.join(BASE_DIR, "leaderboard.json")
LEADERBOARD_KEEP = 20
LEADERBOARD_SHOW = 10
HAND_MODEL_FILE = os.path.join(BASE_DIR, "models", "hand_landmarker.task")
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/latest/hand_landmarker.task")
CRASH_LOG_FILE = os.path.join(BASE_DIR, "crash.log")

# --- Hand tracking ---------------------------------------------------------
TRACKER_INPUT_WIDTH = 640      # frames are downscaled to this before MediaPipe
HAND_DETECTION_CONFIDENCE = 0.5
HAND_PRESENCE_CONFIDENCE = 0.5
HAND_TRACKING_CONFIDENCE = 0.4
BLADE_SMOOTHING = 0.7          # EMA weight of the newest sample (1.0 = no smoothing)
TRAIL_LENGTH = 15              # samples kept in the blade trail
TRAIL_MAX_AGE = 0.30           # seconds before a trail point fully fades
MIN_SLICE_SPEED = 900.0        # px/s (at 720p) the fingertip must move to cut
MAX_SEGMENT_GAP = 0.15         # s; larger gaps between samples are not treated as a swipe
HITBOX_SCALE = 1.3             # hit radius = visible radius * this (forgiving)

# --- Physics ---------------------------------------------------------------
GRAVITY = 1050.0               # px/s^2 at 720p
FRUIT_RADIUS = 52              # px at 720p (watermelon is drawn larger)

# --- Rules -----------------------------------------------------------------
START_LIVES = 3
TIMER_MODE_SECONDS = 60
POINTS_PER_FRUIT = 10
COMBO_WINDOW = 0.5             # max seconds between slices in one combo
TIMER_BOMB_PENALTY = 50        # in timer mode a bomb costs points instead of the round
TIMER_BOMB_ENDS_ROUND = False  # set True to make bombs end timer rounds too

# Survival difficulty ramp (values interpolate from start to end over RAMP_SECONDS)
RAMP_SECONDS = 120.0
SPAWN_INTERVAL = (1.6, 0.65)   # seconds between waves
WAVE_SIZE = (1, 4)             # max fruits per wave
BOMB_CHANCE = (0.10, 0.25)     # chance a wave contains a bomb

# --- Idle / attract mode ---------------------------------------------------
MENU_IDLE_TO_LEADERBOARD = 25.0
LEADERBOARD_IDLE_TO_MENU = 15.0
GAMEOVER_IDLE_AUTOSAVE = 60.0
DEMO_BLADE_AFTER = 2.5         # seconds without a hand before the menu demo blade kicks in

# --- Palette (BGR) ---------------------------------------------------------
NEON_CYAN = (255, 230, 40)
NEON_PINK = (180, 60, 255)
NEON_YELLOW = (40, 230, 255)
NEON_GREEN = (90, 255, 120)
NEON_ORANGE = (30, 150, 255)
NEON_RED = (70, 60, 255)
WHITE = (255, 255, 255)
DARK = (25, 15, 20)
GOLD = (40, 200, 255)
SILVER = (210, 210, 200)
BRONZE = (60, 130, 205)

TRAIL_CORE = (255, 255, 255)
TRAIL_GLOW = (255, 200, 60)
