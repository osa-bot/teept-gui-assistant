import cv2
import os
import requests
import json
from base64 import b64encode


def Google_OCR_makeImageData(imgpath):
    """
    Prepares image data for Google Cloud Vision OCR API request to extract text from UI screenshots.
    
    This method reads an image file, encodes it in base64 format, and structures it into a JSON 
    request format compatible with Google Cloud Vision API. The encoded image is configured for 
    document text detection, enabling accurate extraction of text content from captured UI elements 
    and screen regions during task automation workflows.
    
    Args:
        imgpath (str): The file path to the image file that needs to be processed for text extraction.
    
    Returns:
        bytes: A JSON-encoded byte string containing the formatted API request with the base64-encoded 
            image content and document text detection features configured for optical character recognition.
    """
    with open(imgpath, 'rb') as f:
        ctxt = b64encode(f.read()).decode()
        img_req = {
            'image': {
                'content': ctxt
            },
            'features': [{
                'type': 'DOCUMENT_TEXT_DETECTION',
                # 'type': 'TEXT_DETECTION',
                'maxResults': 1
            }]
        }
    return json.dumps({"requests": img_req}).encode()


def ocr_detection_google(imgpath):
    """
    Extracts text content from an image using Google's Vision API for UI element identification and analysis.
    
    This method enables the system to detect and recognize text within screenshots, which is essential for
    matching user-provided descriptions to UI components. It sends an image to Google's Vision API for text
    detection, constructs a request with the image data, makes a POST request to the API endpoint, and
    processes the response to extract individual text annotations that can be matched against task descriptions.
    
    Args:
        imgpath (str): The file path to the image that will be analyzed for text detection.
    
    Returns:
        list or None: A list of text annotation objects detected in the image, excluding the first
            (full-text) annotation which represents the complete detected text. Returns None if no
            text is detected in the image.
    
    Raises:
        Exception: If the API response does not contain the expected 'responses' field
            or if the API returns an error.
    
    Note:
        The API key used in this method is a placeholder and must be replaced with
        a valid Google Cloud Vision API key obtained from
        https://cloud.google.com/vision before the method can function properly.
    """
    url = 'https://vision.googleapis.com/v1/images:annotate'
    api_key = 'AIzaSyDUc4iOUASJQYkVwSomIArTKhE2C6bHK8U'             # *** Replace with your own Key ***
    imgdata = Google_OCR_makeImageData(imgpath)
    response = requests.post(url,
                             data=imgdata,
                             params={'key': api_key},
                             headers={'Content_Type': 'application/json'})
    # print('*** Text Detection Time Taken:%.3fs ***' % (time.clock() - start))
    print("*** Please replace the Google OCR key at detect_text/ocr.py line 28 with your own (apply in https://cloud.google.com/vision) ***")
    if 'responses' not in response.json():
        raise Exception(response.json())
    if response.json()['responses'] == [{}]:
        # No Text
        return None
    else:
        return response.json()['responses'][0]['textAnnotations'][1:]
