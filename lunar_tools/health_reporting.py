#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import os
import time
from datetime import datetime
import threading

class TelegramBot:
    def __init__(self, token=None, chat_id=None):
        if token is None:
            self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        if chat_id is None:
            self.chat_id = int(os.getenv("TELEGRAM_CHAT_ID"))
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.timeout = 5
        
        
    def send_message(self, message):
        payload = {
            'chat_id': self.chat_id,
            'text': message
        }
        try:
            response = requests.post(f"{self.base_url}/sendMessage", data=payload, timeout=self.timeout)
            if response.json()['ok']:
                print(f"OK! message sent: '{message}'")
            else:
                print(f"FAIL! message failed to send: '{message}' {response.json()}")

        except Exception as e:
            print(f"An error occurred: {e}")

    def get_chat_ids(self):
        try:
            response = requests.get(f"{self.base_url}/getUpdates")
            data = response.json()
            if not data['ok']:
                print(f"Failed to get updates: {data}")
                return []

            chat_info = []
            for update in data['result']:
                if 'message' in update:
                    chat = update['message']['chat']
                    chat_info.append((chat['id'], chat.get('title', 'No Title')))
                elif 'my_chat_member' in update:
                    chat = update['my_chat_member']['chat']
                    chat_info.append((chat['id'], chat.get('title', 'No Title')))
            
            # Remove duplicates by converting to a dictionary and back to a list
            unique_chat_info = list(dict.fromkeys(chat_info))
            return unique_chat_info

        except Exception as e:
            print(f"An error occurred: {e}")
            return []


class HealthReporter:
    def __init__(self, exhibit_name, alive_status_enable_after=60, token=None, chat_id=None, deactivated=False):
        self.disable = deactivated
        if not self.disable:
            self.telegram_bot = TelegramBot(token, chat_id)
            self.register_exhibit(exhibit_name)
            self.maximum_time_without_alive = 10 # seconds
            self.has_been_reported_not_alive = False
            self.last_time_alive = time.time()
            self.start_health_check_thread()
            self.time_started = time.time()
            self.alive_status_enable_after = alive_status_enable_after # to accomodate longer startup times

    def register_exhibit(self, exhibit_name):
        assert exhibit_name is not None
        self.exhibit_name = exhibit_name
        self.send_message(f"STARTED! at {self.get_current_time()}")
    
    def send_message(self, message):
        if not self.disable:
            self.telegram_bot.send_message(f"{self.exhibit_name}: {message}")

    def get_current_time(self):
        return datetime.now().strftime("%H:%M")

    def report_alive(self):
        self.last_time_alive = time.time()

    def start_health_check_thread(self):
        health_check_thread = threading.Thread(target=self.check_alive_status)
        health_check_thread.daemon = True
        health_check_thread.start()

    def report_exception(self, e):
        try:
            exception_message = f"{type(e).__name__}: {e}"
            print(f"report_exception: {exception_message}")
            self.send_message(f"EXCEPTION! {exception_message}")
        except Exception as e:
            print(f"EXCEPTION INSIDE HEALTH REPORTER: {e}")

    def report_message(self, message):
        self.send_message(f"message: {message}")
        

    def check_alive_status(self):
        while True:
            time.sleep(self.maximum_time_without_alive)
            t_now = time.time()
            
            if t_now - self.alive_status_enable_after < self.time_started:
                continue
            
            if t_now - self.last_time_alive > self.maximum_time_without_alive:
                self.send_message("ALARM! Did not receive alive signal.")
                self.last_time_alive = time.time() # So that it will repeat again

if __name__ == "__main__":
    # DO NOT FORGET TO update bashrc/profile with
    # export TELEGRAM_BOT_TOKEN='7191618275:AAGnTVI9eAKYpgl82sM8mWz9fFiIaoQZFc8'
    # export TELEGRAM_CHAT_ID='-4106034916'
    health_reporter = HealthReporter("TESTINSTALL", deactivated=True)
    
    # in while loop, always report that exhibit is alive. This happens in a thread already.
    for i in range(100): #while lopp
        health_reporter.report_alive()
    
    # we can also just throw exceptions in there!
    try: 
        x = y
    except Exception as e:
        health_reporter.report_exception(e)
    
    # or send something friendly!
    health_reporter.report_message("friendly manual message")
    

