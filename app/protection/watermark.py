from blind_watermark import WaterMark

def embed_watermark(input_path: str, output_path: str, payload: str):
    """
    Embed a text payload into the image using DWT-DCT blind watermarking.
    """
    # Initialize with default passwords
    bwm = WaterMark(password_wm=1, password_img=1)
    bwm.read_img(input_path)
    bwm.read_wm(payload, mode='str')
    bwm.embed(output_path)
    return output_path

def extract_watermark(input_path: str, payload_length: int) -> str:
    """
    Extract a text payload from the image.
    """
    bwm = WaterMark(password_wm=1, password_img=1)
    extracted = bwm.extract(input_path, wm_shape=payload_length, mode='str')
    return str(extracted)
