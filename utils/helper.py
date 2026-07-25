"""
---------------------------------------------------------
Helper Functions
---------------------------------------------------------
"""

import os


def create_directory(path):

    os.makedirs(

        path,

        exist_ok=True

    )


def print_header(title):

    print()

    print("="*60)

    print(title)

    print("="*60)


def save_message(message):

    print(message)
