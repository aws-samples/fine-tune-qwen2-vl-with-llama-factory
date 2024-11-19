import base64
from io import BytesIO


def pil_to_base64(pil_image):
    pil_image = pil_image.convert('RGB')
    buffered = BytesIO()
    pil_image.save(buffered, format='JPEG')
    return base64.b64encode(buffered.getvalue()).decode('utf-8')
