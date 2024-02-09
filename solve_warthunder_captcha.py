from io import BytesIO
import numpy as np
import onnxruntime as onr
from PIL import Image




characters_wt = ['2', '3', '4', '5', '6', 
                 '7', '9', 'a', 'b', 'c', 
                 'd', 'e', 'f', 'g', 'h', 
                 'j', 'k', 'm', 'n', 'p', 
                 'q', 'r', 's', 't', 'u', 
                 'v', 'w', 'x', 'y', 'z']
img_width = 150
img_height = 60

ModelWt = onr.InferenceSession('modelwt.onnx')

def solve_task_wt(imgbytes):
    # bytesio actions
    buffered = BytesIO()
    buffered.write(imgbytes)
    # bytesio actions

    # PILLOW actions
    img = Image.open(buffered)
    img = img.resize((img_width, img_height))
    # PILLOW actions

    # numpy actions
    img = np.array(img.convert('L'))
    img = img.astype(np.float32) / 255.
    img = np.expand_dims(img, axis=0)
    img = img.transpose([2,1,0])
    img = np.expand_dims(img, axis=0)
    # numpy actions
    
    # onnxruntime actions
    result_tensor = ModelWt.run(None, {'image': img})[0]
    # onnxruntime actions

    answer, accuracy = get_result(result_tensor, characters_wt, 6)

    return [answer, accuracy]

def test_model():
    import requests
    captcha_image = requests.get(f'https://embed.gaijin.net/captcha').content
    solvation = solve_task_wt(captcha_image)
    # bytesio actions
    buffered = BytesIO()
    buffered.write(captcha_image)
    # bytesio actions

    # PILLOW actions
    img = Image.open(buffered)
    print(f"Answer: {solvation[0]} | Accuracy: {solvation[1]}")
    img.show()

def get_result(pred, characters, max_length):

    accuracy = 1
    last = None
    ans = []

    for item in pred[0]:
        char_ind = item.argmax()

        if char_ind != last and char_ind != 0 and char_ind != len(characters) + 1:
            ans.append(characters[char_ind - 1])
            accuracy *= item[char_ind]

        last = char_ind

    answ = "".join(ans)[:max_length]

    return answ, accuracy

if __name__ == "__main__":
    test_model()