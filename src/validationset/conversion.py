import os
import subprocess

def convert_mp3_to_mkv(source_folder, target_folder):
    # Ensure target folder exists
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # Loop through all files in the source folder
    for file_name in os.listdir(source_folder):
        # Check if the file is an MP3
        if file_name.lower().endswith('.mp3'):
            # Construct the full path to the source and target files
            source_path = os.path.join(source_folder, file_name)
            target_file_name = os.path.splitext(file_name)[0] + '.mkv'
            target_path = os.path.join(target_folder, target_file_name)
            
            # Construct and execute the ffmpeg command
            command = ['ffmpeg', '-i', source_path, '-acodec', 'copy', target_path]
            subprocess.run(command)
    
    print("Conversion complete.")


source_folder = './data/'
target_folder = './data/conv/'
convert_mp3_to_mkv(source_folder, target_folder)
