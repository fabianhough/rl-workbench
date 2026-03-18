from PIL import Image



def save_gif(frames: list, path: str, fps: int=30) -> None:
    '''
    Given a list of rgb arrays (frames), creates a .gif
        at fps and saves to given path

    Parameters:
        frames (list):  List of RGB arrays
        path (str):     String path to save .gif
        fps (int):      Frameratre of .gif    
    '''
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(
        path, save_all=True, append_images=imgs[1:],
        loop=0, duration=1000//fps
    )