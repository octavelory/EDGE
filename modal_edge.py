from modal import Image, App, Volume, gpu
from mediafiredl import MediafireDL as MF
from pydub import AudioSegment
import requests
import os

image = (
   Image.debian_slim(python_version="3.10")
      .apt_install("git", "wget", "wine", "ffmpeg")
      .pip_install("bpy", "accelerate", "wandb", "matplotlib", "soundfile", "librosa", "numpy", "tqdm", "einops", "p-tqdm", "pydub", "scipy", "requests")
      .run_commands("pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html")
      .run_commands("pip install requests mediafiredl fvcore iopath")
      .run_commands('pip install "git+https://github.com/facebookresearch/pytorch3d.git"')
      .run_commands("pip install git+https://github.com/rodrigo-castellon/jukemirlib.git")
      .apt_install("xorg", "libxkbcommon0")
      .run_commands("git clone https://github.com/Stanford-TML/EDGE.git root/EDGE")
      .run_commands("apt-get install xvfb", "Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &", "export DISPLAY=:99")
)
volume = Volume.from_name("dance-models")
app = App(name="dance-generation", image = image)

@app.function(volumes={"/vol": volume})
def load_model(url):
   link = MF.GetFileLink(url)
   print("Downloading model...")
   with open("/vol/checkpoint.pt", "wb") as f:
      f.write(requests.get(link).content)
   print("Download completed")
   volume.commit()

@app.function(volumes={"/vol": volume})
def download_song(spotify_id):
   print(f"Downloading song: {spotify_id}...")
   file_content = requests.get(f"https://musique-maman.vercel.app/download/{spotify_id}").content
   with open("/vol/song.mp3", "wb") as f:
      f.write(file_content)
   song = AudioSegment.from_mp3("/vol/song.mp3")
   song.export("/vol/song.wav", format="wav")
   volume.commit()
   print(os.listdir("/vol"))

@app.function(volumes={"/vol": volume})
def convert_song():
   target = next(filter(lambda x: x.endswith(".mp3"), os.listdir("/vol")))
   os.system(f"ffmpeg -i /vol/{target} -ac 1 /vol/output.wav")
   volume.commit()

@app.function(gpu = gpu.T4(), volumes={"/vol": volume}, timeout=3600)
def generate_dance():
   print("Loading model...")
   os.system("cp /vol/checkpoint.pt EDGE/checkpoint.pt")
   print("Starting inference...")
   # command: python test.py --music_dir "{output_folder}"/ --save_motions --motion_save_dir "{motion_folder}"
   os.system("cd EDGE && python test.py --music_dir /vol --save_motions --motion_save_dir /vol")
   volume.commit()
   print(os.listdir("/vol"))