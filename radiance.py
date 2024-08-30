import os
import anthropic
import subprocess
import re
import concurrent.futures
import cv2
import numpy as np
import assemblyai as aai
import tempfile
from dotenv import load_dotenv
from moviepy.video.compositing.concatenate import concatenate_videoclips
from moviepy.video.io.VideoFileClip import VideoFileClip
from youtube_transcript_api import YouTubeTranscriptApi

os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/Cellar/ffmpeg/7.0.1/bin/ffmpeg"
from moviepy.config import change_settings
change_settings({"FFMPEG_BINARY": "/opt/homebrew/Cellar/ffmpeg/7.0.1/bin/ffmpeg"})
import moviepy.editor as mp
from moviepy.video.tools.subtitles import SubtitlesClip
from moviepy.video.VideoClip import TextClip, ImageClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from fuzzywuzzy import fuzz


class Word:
    text = ""
    start = 0
    end = 0
    confidence = 0
    speaker = None


class TranscriptInterface:
    text = ""
    words = []

load_dotenv()
CONCURRENCY_LIMIT = 5
x_center = 0


def transform_subs(word):
    final_words = []
    split = word.text.split()
    for w in split:
        new = Word()
        new.text = w
        new.start = word.start
        new.end = word.end
        final_words.append(new)
    return final_words

def find_timestamps_of_passage(words, passage, assembled):
    start_timestamp = None
    end_timestamp = None
    if not assembled:
        word_level = list(map(transform_subs, words))
        words = [word for sublist in word_level for word in sublist]
    passage_words = passage.split()
    
    # Iterate over the words in the transcript
    for i in range(len(words) - len(passage_words) + 1):
        match_score = 0
        # Check if the passage matches starting from the current word
        for j in range(len(passage_words)):
            # Use fuzzy matching to compare words
            yt = words[i + j].text.lower()
            claude = passage_words[j].lower()
            similarity = fuzz.ratio(words[i + j].text.lower(), passage_words[j].lower())
            match_score += similarity
        
        # If the average similarity is above a threshold (e.g., 80%), consider it a match
        if match_score / len(passage_words) > 80:
            start_timestamp = words[i].start
            end_timestamp = words[i + len(passage_words) - 1].end
            break
    if start_timestamp is None or end_timestamp is None:
        print("Passage:")
        print(passage)
        print("\nFull Transcript:")
        print(" ".join(word.text for word in words))
    return start_timestamp, end_timestamp


def transcribe_audio(FILE_URL):
    api_key = os.getenv('ASSEMBLY_API_KEY')
    aai.settings.api_key = api_key
    config = aai.TranscriptionConfig(
      speaker_labels=False,
      speakers_expected=1
    )
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(
      FILE_URL,
      config=config
    )
    return FILE_URL, transcript

# Sort files by part number
def get_part_number(filename):
    match = re.search(r'_part_(\d+)', filename)
    return int(match.group(1)) if match else 0


# Function to adjust timestamps
def adjust_timestamps(transcripts, durations):
    cumulative_duration = 0
    for i, transcript in enumerate(transcripts):
        for word in transcript.words:
            word.start += cumulative_duration
            word.end += cumulative_duration
        cumulative_duration += durations[i]


# Function to get audio duration
def get_audio_duration(filename):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    return float(result.stdout)


# Function to split audio and return durations
def split_audio(filename, k, TEMP_DIR, parts_dir):
    duration = get_audio_duration(os.path.join(TEMP_DIR, filename))
    segment_duration = duration / k
    base_name, ext = os.path.splitext(filename)
    durations = []
    filename = os.path.join(TEMP_DIR, filename)
    for i in range(k):
        start_time = i * segment_duration
        output_filename = os.path.join(parts_dir, f"{base_name}_part_{i + 1}{ext}")
        print(output_filename)
        subprocess.run([
            "ffmpeg", "-i", filename, "-ss", str(start_time), "-t", str(segment_duration),
            "-c", "copy", output_filename
        ])
        durations.append(segment_duration * 1000)  # Store duration in milliseconds

    return durations


def download_video_audio_parallel(youtube_url, TEMP_DIR):
    output_dir = os.path.join(TEMP_DIR, "%(title)s.%(ext)s")

    # Download audio
    #"res:720,fps:30,+br"
    #+size,+br
    result = subprocess.run(
        ["yt-dlp", "-S", "res:720,fps:30,+br", "--extract-audio", "-k", "-o", output_dir, youtube_url],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    try:
        audio_filename = result.stdout.decode().split("[ExtractAudio] Destination: " + str(TEMP_DIR + "/"))[1].split("\n")[0]
        video_file_url = result.stdout.decode().split('Merging formats into "')[1].split('"\n')[0]
    except:
        print("Error: " + result.stdout.decode().split("Error")[0])

    extension = audio_filename.split(".")[1]
    parts_dir = os.path.join(TEMP_DIR, 'parts')
    os.makedirs(parts_dir)
    # Split audio into CONCURRENCY_LIMIT fragments and return durations
    return split_audio(audio_filename, CONCURRENCY_LIMIT, TEMP_DIR, parts_dir), extension, video_file_url


def transcribe_audio_parallel(TEMP_DIR, audio_extension, durations):
    directory_path = os.path.join(TEMP_DIR, 'parts')
    files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith(audio_extension)]
    files.sort(key=get_part_number)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit tasks returning (filename, transcript) tuple
        futures = executor.map(transcribe_audio, files)

        ts = []
        for file, transcript in futures:
            try:
                ts.append(transcript)
            except Exception as e:
                print(f"Error transcribing {file}: {e}")

    adjust_timestamps(ts, durations)
    return ts


def fuse_transcript(transcripts):
    final_transcript = TranscriptInterface()
    for t in transcripts:
        final_transcript.text += t.text
        final_transcript.words += t.words
    return final_transcript


def get_highlights(transcript):
    api_key = os.getenv('ANTHROPIC_API_KEY')
    client = anthropic.Anthropic(api_key=api_key)
    prompt = "Please give me the 4 most interesting text excerpts of the given transcript. For each excerpt, provide a short, catchy title for social media and a very brief description. It is important that you return the transcript excerpts written in the exact same way as it was given to you. Transcript: " + transcript.text
    message = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1024,
        system="You are my social media adviser. I will provide you with a transcript and I want you to filter out the most interesting / potentially viral parts of a text to then later use as my social media clips. It is very important that you don't truncate the text or edit the text excerpts in any way.",
        tools=[
            {
                "name": "get_highlights",
                "description": "Get the most interesting excerpts of a transcript in order to create social media short form content",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "highlight_1": {
                            "type": "object",
                            "properties": {
                                "excerpt": {"type": "string", "description": "The first excerpt of the transcript"},
                                "title": {"type": "string", "description": "A short, catchy title for the excerpt"},
                                "description": {"type": "string", "description": "A very brief description of the excerpt"}
                            },
                            "required": ["excerpt", "title", "description"]
                        },
                        "highlight_2": {
                            "type": "object",
                            "properties": {
                                "excerpt": {"type": "string", "description": "The second excerpt of the transcript"},
                                "title": {"type": "string", "description": "A short, catchy title for the excerpt"},
                                "description": {"type": "string", "description": "A very brief description of the excerpt"}
                            },
                            "required": ["excerpt", "title", "description"]
                        },
                        "highlight_3": {
                            "type": "object",
                            "properties": {
                                "excerpt": {"type": "string", "description": "The third excerpt of the transcript"},
                                "title": {"type": "string", "description": "A short, catchy title for the excerpt"},
                                "description": {"type": "string", "description": "A very brief description of the excerpt"}
                            },
                            "required": ["excerpt", "title", "description"]
                        },
                        "highlight_4": {
                            "type": "object",
                            "properties": {
                                "excerpt": {"type": "string", "description": "The fourth excerpt of the transcript"},
                                "title": {"type": "string", "description": "A short, catchy title for the excerpt"},
                                "description": {"type": "string", "description": "A very brief description of the excerpt"}
                            },
                            "required": ["excerpt", "title", "description"]
                        },
                    },
                    "required": ["highlight_1", "highlight_2", "highlight_3", "highlight_4"],
                },
            }
        ],
        tool_choice={"type": "tool", "name": "get_highlights"},
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return message


def edit_video(start, end, video_file_url):
    global x_center
    clip = mp.VideoFileClip(video_file_url)
    video = clip.subclip(start / 1000, end / 1000)
    # Define the desired aspect ratio
    target_aspect_ratio = 9 / 16
    # Get the current dimensions of the video
    original_width, original_height = video.size
    x_center = original_width / 2
    new_width = int(original_height * target_aspect_ratio) + 1
    new_height = original_height
    # Load pre-trained YOLO model
    net = cv2.dnn.readNet("./models/yolo/yolov3.weights", "./models/yolo/yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    def crop_video_person(get_frame, t):
        global x_center
        frame = get_frame(t)
        # Detect objects in the frame using YOLO
        if np.round(t * 30) % 0 == 0:  # Update the crop pos every % frames
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if class_id == 0 and confidence > 0.5:  # Class ID 0 corresponds to 'person'
                        # Extract bounding box coordinates
                        x_center = int(detection[0] * frame.shape[1])
                        break

        new_width = int(original_height * target_aspect_ratio) + 1
        x1 = max(0, x_center - new_width / 2)
        x2 = min(original_width, x_center + new_width / 2)
        # Ensure crop box does not exceed video bounds
        if x2 - x1 < new_width:
            if x1 == 0:
                x2 = new_width
            elif x2 == original_width:
                x1 = original_width - new_width
        return frame[:, int(x1):int(x2)]

    # Process the video
    processed_video = video.fl(lambda gf, t: crop_video_person(gf, t))

    return processed_video, new_width, new_height


def create_srt_file(subtitles, output_srt_file):
    """
    Create a SubRip (.srt) file from the given subtitles and write it to the specified output file.

    Args:
    - subtitles (list): A list of tuples containing subtitle information (start time in milliseconds,
                        end time in milliseconds, text).
    - output_srt_file (str): The path to the output .srt file.
    """
    # Open the output .srt file in write mode
    with open(output_srt_file, "w") as f:
        # Initialize a counter for subtitle indices
        subtitle_index = 1

        # Iterate over each subtitle in the subtitles list
        for index, (start_ms, end_ms, text) in enumerate(subtitles):
            # Convert milliseconds to seconds
            start = start_ms / 1000
            end = end_ms / 1000

            # Write the subtitle index
            f.write(str(subtitle_index) + "\n")

            # Write the subtitle timing in the format: HH:MM:SS,sss --> HH:MM:SS,sss
            f.write("{:02d}:{:02d}:{:06.3f} --> {:02d}:{:02d}:{:06.3f}\n".format(
                int(start // 3600), int((start % 3600) // 60), start % 60,
                int(end // 3600), int((end % 3600) // 60), end % 60
            ).replace('.', ','))

            # Write the subtitle text
            f.write(text + "\n\n")

            # Increment the subtitle index for the next subtitle
            subtitle_index += 1

    print("Subtitles saved to:", output_srt_file)
    return output_srt_file


def create_subtitles(transcript, start, end, output_dir, video_title, ):
    # Might miss the last word when assembled
    relevant_words = [word for word in transcript.words if start <= word.start < end]
    subtitles = [(word.start - start, word.end - start, word.text) for word in relevant_words]
    subtitle_title = video_title.split(".")[0] + "_subtitles.srt"
    output_srt_file = os.path.join(output_dir, subtitle_title)
    return create_srt_file(subtitles, output_srt_file)


def write_subtitles_to_video(video, subtitle_url, w, h):
    screensize = [w , h // 5]
    generator = lambda txt: TextClip(txt, font='Proxima-Nova-Semibold',size = screensize, method='caption',
                                     color='white', stroke_color="black", stroke_width=1,)
    subs = SubtitlesClip(subtitle_url, generator)
    subtitles = SubtitlesClip(subs, generator)
    result = CompositeVideoClip([video, subtitles.set_pos(('center', 'bottom'))])
    return result

def get_youtube_captions(youtube_url):
    vid_id = youtube_url.split("youtube.com/watch?v=")[1]
    try:
        yt_ts = YouTubeTranscriptApi.get_transcript(vid_id)
    except:
        return None
    t = TranscriptInterface()
    for i, ts in enumerate(yt_ts):
        t.text += ts["text"] + " "
        word = Word()
        word.text = ts["text"]
        word.start = np.round(float(ts["start"]) * 1000)
        if i == len(yt_ts) - 1:
            word.end = np.round(float(ts["start"] + ts["duration"]) * 1000)
        else:
            word.end = np.round(float(yt_ts[i + 1]["start"]) * 1000)
        t.words.append(word)
    return t


def add_outro(video):
    outro = VideoFileClip("./scaled_outro.mov")
    # Concatenate the clips
    return concatenate_videoclips([video, outro])


def add_watermark(video):
    watermark = ImageClip("./icon.png")

    watermark = watermark.resize(width=watermark.size[0] // 2, height=watermark.size[1] // 2)
    # Set the position of the watermark to the top right corner with a margin of 10 pixels
    watermark_position = (video.size[0] - watermark.size[0] - 10, 10)
    # Set the duration of the watermark to match the video
    watermark = watermark.set_duration(video.duration)

    # Overlay the watermark on the video
    return CompositeVideoClip([video, watermark.set_position(watermark_position).set_opacity(0.5)])

def extract_xml_content(xml_string, tag):
    pattern = f'<{tag}>(.*?)</{tag}>'
    match = re.search(pattern, xml_string, re.DOTALL)
    return match.group(1).strip() if match else ''

def generate_clip(youtube_url):
    with tempfile.TemporaryDirectory() as TEMP_DIR:
        # Download Video / Audio
        print("Downloading Video")
        durations, extension, temp_video_url = download_video_audio_parallel(youtube_url, TEMP_DIR)

        # Transcribe Audio
        print("Transcribing Audio")
        assembled = False
        transcript = get_youtube_captions(youtube_url)
        if transcript is None:
            print("No Yt Captions found")
            assembled = True
            transcript = fuse_transcript(transcribe_audio_parallel(TEMP_DIR, extension, durations))

        # Generate Highlights
        print("Generating Highlights")
        message = get_highlights(transcript)

        # Make Clip
        print("Editing Clips")
        for highlight in message.content[-1].input:
            highlight_data = message.content[-1].input[highlight]
            # Extract title, description, and excerpt from the XML-like string
            title = extract_xml_content(highlight_data, 'title')
            description = extract_xml_content(highlight_data, 'description')
            excerpt = extract_xml_content(highlight_data, 'excerpt')

            start, end = find_timestamps_of_passage(transcript.words, excerpt, assembled)
            if start is None or end is None:
                print("Error From Claude :/")
                continue
            video, w, h = edit_video(start, end, temp_video_url)

            video_title = highlight + "_" + temp_video_url.split(TEMP_DIR + "/")[1]
            final_dir_path = '/Users/ericnazarenus/Desktop/Rickberd/Development/Web Dev/Portfolio/nazarenus_backend/clips'
            final_dir = os.path.join(final_dir_path, video_title.split(".")[0].split("_")[2])
            video_output_url = os.path.join(final_dir, video_title)
            os.makedirs(final_dir, exist_ok=True)
            # Add Subtitles
            print("Adding Subtitles")
            subtitle_url = create_subtitles(transcript, start, end, final_dir, video_title)
            video = write_subtitles_to_video(video, subtitle_url, w, h)

            # Add Outro
            print("Adding Outro")
            video = add_outro(video)
            video = add_watermark(video)
            video.write_videofile(os.path.join(final_dir, video_output_url), codec='libx264', audio_codec='aac')

            print("Saved Clip to: " + os.path.join(final_dir, video_output_url))
            yield os.path.join(final_dir, video_output_url), title, description

if __name__ == '__main__':
    generate_clip("https://www.youtube.com/watch?v=kbWLE07NGcw")
