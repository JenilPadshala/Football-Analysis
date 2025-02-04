from utils import read_video, save_video
from trackers import Tracker

def main():
    # Read video
    video_frames = read_video("input_videos/08fd33_4.mp4")

    # Initialize tracker
    tracker = Tracker("models/best.pt")

    # Get object tracks
    tracks = tracker.get_object_tracks(video_frames)

    #save video
    save_video(video_frames, "output_videos/output.avi")

if __name__ == "__main__":
    main()