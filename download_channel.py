from pytubefix import Channel
import os
import re
import os
from faster_whisper import WhisperModel

def download_audio_from_youtube_channel(channel_url, output_path, max_videos=10, max_duration=1800):

    # Function to sanitize video title for file naming
    def sanitize_filename(filename):
        return re.sub(r'[\\/*?:"<>|]', "", filename)

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Create a Channel object
    yt_channel = Channel(channel_url)

    # Loop through all videos in the channel
    for video in yt_channel.videos[:max_videos]:
        # Convert the video duration to seconds
        print(f"Nombre del video: {video.title}\tDuración: {video.length}")
        # duration_seconds = duration_to_seconds(video.length)

        # Check if the video is shorter than 0.5 hour (1800 seconds)
        if video.length <= max_duration:
            # Sanitize the video title to create a safe filename
            sanitized_title = sanitize_filename(video.title)

            if os.path.exists(os.path.join(output_path, sanitized_title + ".mp3")):
                print(f"Audio has already been downloaded, jumping to next audio...")
                continue

            # Download only the audio track as an MP3 file
            audio_stream = video.streams.filter(only_audio=True).first()
            if audio_stream:
                audio_stream.download(output_path=output_path, filename=sanitized_title + ".mp3")
                print(f"Downloaded: {sanitized_title}.mp3")
            else:
                print(f"No audio stream found for video: {video.title}")
        else:
            print(f"Skipped (longer than {(max_duration/60):.4f} hours): {video.title}")

    print("All eligible videos downloaded.")


# channel_url = "https://www.youtube.com/@BorjaBandera/videos"
# output_path = "/content/audiosBorjaBandera"
# download_audio_from_youtube_channel(channel_url, output_path, max_videos=30, max_duration=1800)



def transcribe_audios_from_folder(audio_folder="audiosBorjaBandera/",
                                  transcription_folder = "transcriptionsBorjaBandera/",
                                  model_name="Systran/faster-whisper-small",
                                  device="cuda"):

    # Define the model and the folder containing the audio files
    model = WhisperModel(model_name, device=device, compute_type="float16")
    # Ensure the folders exist
    if not os.path.exists(audio_folder):
        print(f"The folder '{audio_folder}' does not exist.")
        exit(1)

    os.makedirs(transcription_folder, exist_ok=True)

    # Loop through all audio files in the folder
    for audio_file in os.listdir(audio_folder):
        if audio_file.endswith(".mp3"):  # Process only .mp3 files
            print(f"Transcribing file {audio_file} ...")
            audio_path = os.path.join(audio_folder, audio_file)

            # print(os.path.join(transcription_folder, audio_file + ".txt"))
            # print(transcription_folder)
            # print(audio_file.split(".")[0] + ".txt")
            # print()
            if os.path.exists(os.path.join(transcription_folder, audio_file.split(".")[0] + ".txt")):
                print(f"Audio has already been transcribed, jumping to next audio...")
                continue

            # Transcribe the audio file
            segments, info = model.transcribe(audio_path, beam_size=5, language="es")

            # Combine the transcribed segments into a single string
            transcription = ""
            for segment in segments:
                transcription += segment.text
            transcription = transcription.strip()
            # Save the transcription to a .txt file
            transcription_filename = os.path.splitext(audio_file)[0] + ".txt"
            transcription_path = os.path.join(transcription_folder, transcription_filename)

            with open(transcription_path, "w", encoding="utf-8") as f:
                f.write(transcription)

            print(f"Transcription saved to {transcription_path}")


# audio_folder = "audiosBorjaBandera/"
# transcription_folder = "transcriptionsBorjaBandera/"
# transcribe_audios_from_folder(audio_folder, transcription_folder)




def download_and_transcribe_videos_from_youtube_channel(channel_url, output_path, max_videos=10, max_duration=1800,
                                                       transcription_folder="transcriptionsBorjaBandera/",
                                                       audio_folder="audiosBorjaBandera/",
                                                       model_name="Systran/faster-whisper-small",
                                                       device="cuda"):

    # Function to sanitize video title for file naming
    def sanitize_filename(filename):
        return re.sub(r'[\\/*?:"<>|]', "", filename)

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(transcription_folder, exist_ok=True)
    os.makedirs(audio_folder, exist_ok=True)

    # Create a WhisperModel instance for transcription
    model = WhisperModel(model_name, device=device, compute_type="float16")

    # Create a Channel object
    yt_channel = Channel(channel_url)

    # Loop through all videos in the channel
    for video in yt_channel.videos[:max_videos]:
        # Convert the video duration to seconds
        print(f"Nombre del video: {video.title}\tDuración: {video.length}")

        # Check if the video is shorter than max_duration seconds
        if video.length <= max_duration:
            # Sanitize the video title to create a safe filename
            sanitized_title = sanitize_filename(video.title)

            # Download only the audio track as an MP3 file
            if not os.path.exists(os.path.join(output_path, sanitized_title + ".mp3")):
                audio_stream = video.streams.filter(only_audio=True).first()
                if audio_stream:
                    audio_stream.download(output_path=output_path, filename=sanitized_title + ".mp3")
                    print(f"Downloaded: {sanitized_title}.mp3")
                else:
                    print(f"No audio stream found for video: {video.title}")

            # Transcribe the downloaded audio file
            audio_file_path = os.path.join(output_path, sanitized_title + ".mp3")
            if os.path.exists(audio_file_path):
                if not os.path.exists(os.path.join(transcription_folder, sanitized_title + ".txt")):
                    print(f"Transcribing file {sanitized_title}.mp3 ...")
                    segments, info = model.transcribe(audio_file_path, beam_size=5, language="es")

                    # Combine the transcribed segments into a single string
                    transcription = ""
                    for segment in segments:
                        transcription += segment.text
                    transcription = transcription.strip()

                    # Save the transcription to a .txt file
                    transcription_path = os.path.join(transcription_folder, sanitized_title + ".txt")
                    with open(transcription_path, "w", encoding="utf-8") as f:
                        f.write(transcription)

                    print(f"Transcription saved to {transcription_path}")
                else:
                    print(f"Audio {sanitized_title}.mp3 has already been transcribed, skipping...")
            else:
                print(f"Audio file {sanitized_title}.mp3 not found, skipping transcription.")

        else:
            print(f"Skipped (longer than {(max_duration / 60):.4f} hours): {video.title}")

    print("All eligible videos downloaded and transcribed.")