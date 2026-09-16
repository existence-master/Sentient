# Sentient glasses firmware (ESP32-S3). Build with ESP-IDF v5.3 or newer:
#   idf.py set-target esp32s3 && idf.py build flash monitor
cmake_minimum_required(VERSION 3.16)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(sentient_glasses)
