from PIL import Image



def save_gif(frames: list, path: str, fps: int=30):
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(
        path, save_all=True, append_images=imgs[1:],
        loop=0, duration=1000//fps
    )